import json
import csv
import time
import base64
import asyncio
import os
from datetime import datetime
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, WebSocket, Query, Request
from fastapi.responses import Response

from signalwire.rest import Client
from openai import AsyncOpenAI
import websockets

SIGNALWIRE_PROJECT_ID = os.getenv("SIGNALWIRE_PROJECT_ID")
SIGNALWIRE_TOKEN = os.getenv("SIGNALWIRE_TOKEN")
SIGNALWIRE_SPACE_URL = os.getenv("SIGNALWIRE_SPACE_URL")
SIGNALWIRE_FROM_NUMBER = os.getenv("SIGNALWIRE_FROM_NUMBER")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PUBLIC_HOST = os.getenv("PUBLIC_HOST")

STT_MODEL = "nova-2-phonecall"
TTS_MODEL = "aura-luna-en"
ENCODING = "mulaw"
SAMPLE_RATE = 8000

SILENCE_TIMEOUT = 1.2
MAX_INITIAL_SILENCE = 6.0
MAX_CALL_DURATION = 420
MAX_CONCURRENT_CALLS = 3

SCRIPT_FILE = "script.txt"
LEADS_FILE = "leads.csv"
RESULTS_FILE = "results.csv"

VOICEMAIL_KEYWORDS = [
    "leave a message", "voicemail", "after the beep",
    "record your message", "at the tone",
    "not available", "unavailable", "please leave"
]

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
sw_client = Client(
    SIGNALWIRE_PROJECT_ID,
    SIGNALWIRE_TOKEN,
    signalwire_space_url=SIGNALWIRE_SPACE_URL
)

app = FastAPI(title="VoiceBot API", version="1.0.0")
call_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

def load_script():
    if os.path.exists(SCRIPT_FILE):
        with open(SCRIPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "You are Anna. Keep responses short and friendly."

SYSTEM_PROMPT = load_script()

def load_leads():
    if not os.path.exists(LEADS_FILE):
        return []
    with open(LEADS_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

LEADS = load_leads()

def save_result(phone, name, transcript, label):
    exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "phone", "name", "label", "transcript"])
        writer.writerow([datetime.utcnow().isoformat(), phone, name, label, transcript])

def normalize_phone(phone):
    phone = phone.strip().replace(" ", "").replace("-", "")
    if not phone.startswith("+"):
        phone = "+1" + phone
    return phone

async def classify_call(transcript):
    if not transcript:
        return "no_answer"

    lower = transcript.lower()
    for kw in VOICEMAIL_KEYWORDS:
        if kw in lower:
            return "voicemail"

    try:
        res = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": transcript[:800]}],
            temperature=0,
            max_tokens=5
        )
        return res.choices[0].message.content.strip().lower()
    except:
        return "error"

@app.post("/call-all")
async def call_all():
    for lead in LEADS:
        phone = normalize_phone(lead.get("phone", ""))
        name = lead.get("name", "there")

        sw_client.calls.create(
            to=phone,
            from_=SIGNALWIRE_FROM_NUMBER,
            url=f"https://{PUBLIC_HOST}/voice?name={quote(name)}&phone={quote(phone)}"
        )
        await asyncio.sleep(1)

    return {"status": "batch started"}

@app.post("/voice")
async def voice(request: Request):
    q = parse_qs(request.url.query)
    name = q.get("name", ["there"])[0]
    phone = q.get("phone", ["unknown"])[0]

    if call_semaphore._value <= 0:
        save_result(phone, name, "", "busy")
        return Response(
            content="<Response><Reject reason='busy'/></Response>",
            media_type="application/xml",
            headers={"Content-Type": "application/xml"}
        )

    stream_url = f"wss://{PUBLIC_HOST}/media?name={quote(name)}&phone={quote(phone)}"

    return Response(
        content=f"""
<Response>
  <Connect>
    <Stream url="{stream_url}" />
  </Connect>
</Response>
""",
        media_type="application/xml",
        headers={"Content-Type": "application/xml"}
    )

@app.websocket("/media")
async def media(ws: WebSocket, name: str = Query("there"), phone: str = Query("unknown")):
    await ws.accept()
    await call_semaphore.acquire()

    stream_sid = None
    transcript_log = []
    conversation = [{"role": "system", "content": SYSTEM_PROMPT.replace("{name}", name)}]

    dg_stt = await websockets.connect(
        f"wss://api.deepgram.com/v2/listen?encoding={ENCODING}&sample_rate={SAMPLE_RATE}&model={STT_MODEL}&endpointing=true",
        extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    )

    dg_tts = await websockets.connect(
        f"wss://api.deepgram.com/v2/speak?model={TTS_MODEL}&encoding={ENCODING}&sample_rate={SAMPLE_RATE}",
        extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    )

    async def wait_for_stream():
        nonlocal stream_sid
        while stream_sid is None:
            await asyncio.sleep(0.01)

    async def speak(text):
        await wait_for_stream()
        await dg_tts.send(json.dumps({"type": "Speak", "text": text}))
        await dg_tts.send(json.dumps({"type": "Flush"}))
        while True:
            msg = await dg_tts.recv()
            if isinstance(msg, bytes):
                await ws.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(msg).decode()}
                }))
            else:
                break

    asyncio.create_task(speak(f"Hi {name}, this is Anna. Can you hear me okay?"))

    start_time = time.time()
    last_user_speech = time.time()

    try:
        while True:
            if time.time() - start_time > MAX_CALL_DURATION:
                break

            data = await ws.receive_json()
            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]

            elif event == "media":
                audio = base64.b64decode(data["media"]["payload"])
                await dg_stt.send(audio)

            elif event == "stop":
                break

            if time.time() - last_user_speech > MAX_INITIAL_SILENCE:
                break

    finally:
        await dg_stt.close()
        await dg_tts.close()
        transcript = "\n".join(transcript_log)
        label = await classify_call(transcript)
        save_result(phone, name, transcript, label)
        call_semaphore.release()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
