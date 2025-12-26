# =========================
# STANDARD LIBRARIES
# =========================
import json
import csv
import time
import base64
import asyncio
import os
from datetime import datetime
from urllib.parse import parse_qs

# =========================
# FASTAPI & WEB SOCKETS
# =========================
from fastapi import FastAPI, WebSocket, Query, Request
from fastapi.responses import Response

# =========================
# THIRD-PARTY SDKs
# =========================
from signalwire.rest import Client
from openai import AsyncOpenAI
import websockets

# ======================================================
# CONFIG & LIMITS
# ======================================================

SIGNALWIRE_PROJECT_ID = os.getenv("SIGNALWIRE_PROJECT_ID")
SIGNALWIRE_TOKEN = os.getenv("SIGNALWIRE_TOKEN")
SIGNALWIRE_SPACE_URL = os.getenv("SIGNALWIRE_SPACE_URL")
SIGNALWIRE_FROM_NUMBER = os.getenv("SIGNALWIRE_FROM_NUMBER")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PUBLIC_HOST = os.getenv("PUBLIC_HOST")

# =========================
# AUDIO CONFIG
# =========================
STT_MODEL = "nova-2-phonecall"
TTS_MODEL = "aura-luna-en"

ENCODING = "mulaw"
SAMPLE_RATE = 8000

# =========================
# CALL CONTROL LIMITS
# =========================
SILENCE_TIMEOUT = 1.2
MAX_INITIAL_SILENCE = 6.0
MAX_CALL_DURATION = 420
MAX_CONCURRENT_CALLS = 3

# =========================
# FILES
# =========================
SCRIPT_FILE = "script.txt"
LEADS_FILE = "leads.csv"
RESULTS_FILE = "results.csv"

# =========================
# VOICEMAIL DETECTION
# =========================
VOICEMAIL_KEYWORDS = [
    "leave a message", "voicemail", "not available", "after the beep",
    "record your message", "at the tone", "unavailable",
    "can't answer", "please leave", "recording"
]

# ======================================================
# CLIENTS & GLOBALS
# ======================================================

# Initialize clients lazily to avoid crashes on import
openai_client = None
sw_client = None

def get_openai_client():
    global openai_client
    if openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return openai_client

def get_signalwire_client():
    global sw_client
    if sw_client is None:
        if not all([SIGNALWIRE_PROJECT_ID, SIGNALWIRE_TOKEN, SIGNALWIRE_SPACE_URL]):
            raise ValueError("SignalWire environment variables are not set")
        sw_client = Client(
            SIGNALWIRE_PROJECT_ID,
            SIGNALWIRE_TOKEN,
            signalwire_space_url=SIGNALWIRE_SPACE_URL
        )
    return sw_client

app = FastAPI(title="VoiceBot API", version="1.0.0")
call_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all exceptions and return proper error response"""
    return {
        "error": "Internal server error",
        "message": str(exc),
        "path": str(request.url.path)
    }, 500

# ======================================================
# ENVIRONMENT VALIDATION
# ======================================================

def validate_env():
    """Validate environment variables. Returns list of missing vars."""
    required_vars = [
        "SIGNALWIRE_PROJECT_ID",
        "SIGNALWIRE_TOKEN", 
        "SIGNALWIRE_SPACE_URL",
        "SIGNALWIRE_FROM_NUMBER",
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "PUBLIC_HOST"
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    return missing

@app.on_event("startup")
async def startup_event():
    """Validate environment on startup and log warnings."""
    missing = validate_env()
    if missing:
        print(f"⚠️  Warning: Missing environment variables: {', '.join(missing)}")
        print("⚠️  The app will start but some features may not work.")
    else:
        print("✅ All required environment variables are set.")

# ======================================================
# DATA PERSISTENCE
# ======================================================

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
        reader = csv.DictReader(f)
        leads = []
        for row in reader:
            normalized = {k.lower(): v for k, v in row.items()}
            leads.append(normalized)
        return leads

LEADS = load_leads()

def save_result(phone, name, transcript, label):
    exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "phone", "name", "label", "transcript"])
        writer.writerow([datetime.utcnow().isoformat(), phone, name, label, transcript])

# ======================================================
# AI HELPERS
# ======================================================

async def classify_call(transcript):
    if not transcript:
        return "no_answer"
    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Return only ONE word:\n"
                    "interested, not_interested, voicemail, angry, no_answer, error\n\n"
                    f"{transcript[:800]}"
                )
            }],
            temperature=0,
            max_tokens=5
        )
        return response.choices[0].message.content.strip().lower()
    except:
        return "error"

# ======================================================
# WEBHOOKS & ROUTES
# ======================================================

@app.post("/call-all")
async def call_all_post():
    """Start calling all leads (POST)"""
    return await call_all()

@app.get("/call-all")
async def call_all_get():
    """Start calling all leads (GET - for browser testing)"""
    return await call_all()

def validate_phone_number(phone):
    """Validate and normalize phone number to E.164 format"""
    if not phone:
        return None
 
    phone = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")

    if not phone.startswith("+"):
        if phone.startswith("1") and len(phone) == 11:
            phone = "+" + phone
        elif len(phone) == 10:
            phone = "+1" + phone  # Default to US country code
        else:
            phone = "+" + phone
    return phone

async def call_all():
    """Start calling all leads"""
    try:
        client = get_signalwire_client()
        if not PUBLIC_HOST:
            return {"error": "PUBLIC_HOST environment variable is not set"}, 400
        if not LEADS:
            return {"error": "No leads found in leads.csv", "leads_count": 0}, 400
        
       
        if not SIGNALWIRE_FROM_NUMBER:
            return {"error": "SIGNALWIRE_FROM_NUMBER environment variable is not set"}, 400
        
        from_number = validate_phone_number(SIGNALWIRE_FROM_NUMBER)
        if not from_number:
            return {
                "error": "SIGNALWIRE_FROM_NUMBER must be in E.164 format (e.g., +1234567890)",
                "current_value": SIGNALWIRE_FROM_NUMBER
            }, 400
        
        call_count = 0
        errors = []
        for lead in LEADS:
            try:
               
                phone = lead.get("phone") or lead.get("Phone") or ""
                name = lead.get("name") or lead.get("Name") or "there"
                
                if not phone:
                    errors.append(f"Lead missing phone number: {lead}")
                    continue
                
          
                normalized_phone = validate_phone_number(phone)
                if not normalized_phone:
                    errors.append(f"Invalid phone number format: {phone}")
                    continue
                
                client.calls.create(
                    to=normalized_phone,
                    from_=from_number,
                    url=f"https://{PUBLIC_HOST}/voice?name={name}&phone={normalized_phone}"
                )
                call_count += 1
                await asyncio.sleep(1)
            except Exception as e:
                error_msg = f"Error calling {lead.get('phone') or lead.get('Phone', 'unknown')}: {e}"
                print(error_msg)
                errors.append(error_msg)
                continue
        
        result = {
            "status": "batch started",
            "total_leads": len(LEADS),
            "calls_initiated": call_count,
            "message": f"Started calling {call_count} leads"
        }
        if errors:
            result["errors"] = errors
        return result
    except ValueError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        return {"error": f"Failed to start calls: {str(e)}"}, 500

@app.post("/voice")
async def voice(request: Request):
    q = parse_qs(request.url.query)
    name = q.get("name", ["there"])[0]
    phone = q.get("phone", ["unknown"])[0]
    
    # Clean up phone number
    phone = phone.strip()
    if not phone.startswith("+"):
        phone = "+" + phone.lstrip("+")
    
    print(f"📞 Incoming call: {name} at {phone}")

    if call_semaphore._value <= 0:
        print(f"❌ Call rejected: Too many concurrent calls")
        save_result(phone, name, "", "busy")
        return Response(
            content='<Response><Reject reason="busy" /></Response>',
            media_type="application/xml"
        )

    from urllib.parse import quote
 
    encoded_name = quote(name)
    encoded_phone = quote(phone)
    stream_url = f"wss://{PUBLIC_HOST}/media?name={encoded_name}&phone={encoded_phone}"
    print(f"✅ Accepting call, connecting to: {stream_url}")
    

    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""
    
    return Response(
        content=xml_response,
        media_type="application/xml"
    )

# ======================================================
# WEBSOCKET STREAM HANDLER
# ======================================================

@app.websocket("/media")
async def media(ws: WebSocket, name: str = Query("there"), phone: str = Query("unknown")):
    print(f"🔌 WebSocket connection attempt: {name} at {phone}")
    await call_semaphore.acquire()
    

    dg_stt = None
    dg_tts = None
    transcript_log = []
    
    try:
        await ws.accept()
        print(f"✅ WebSocket connected: {name} at {phone}")
        stream_sid = None

        conversation = [{
            "role": "system",
            "content": SYSTEM_PROMPT.replace("{name}", name)
        }]
        audio_out_queue = asyncio.Queue()
        user_speech_queue = asyncio.Queue()

        call_active = True
        speaking_task = None
        last_user_speech_time = time.time()
        call_start_time = time.time()
        last_ai_turn_time = 0

        dg_stt = await websockets.connect(
            f"wss://api.deepgram.com/v2/listen"
            f"?encoding={ENCODING}&sample_rate={SAMPLE_RATE}&model={STT_MODEL}"
            f"&interim_results=true&utterance_end_ms=1000&endpointing=true",
            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        )

        dg_tts = await websockets.connect(
            f"wss://api.deepgram.com/v2/speak"
            f"?model={TTS_MODEL}&encoding={ENCODING}&sample_rate={SAMPLE_RATE}",
            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        )

        async def speak_response(text):
            await dg_tts.send(json.dumps({"type": "Speak", "text": text}))
            await dg_tts.send(json.dumps({"type": "Flush"}))
            while True:
                msg = await dg_tts.recv()
                if isinstance(msg, bytes):
                    payload = base64.b64encode(msg).decode("utf-8")
                    await ws.send_text(json.dumps({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": payload}
                    }))
                else:
                    break

        async def generate_and_speak():
            nonlocal last_ai_turn_time
            client = get_openai_client()
            res = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=conversation,
                temperature=0.7,
                max_tokens=75
            )
            ai_text = res.choices[0].message.content.strip()
            conversation.append({"role": "assistant", "content": ai_text})
            transcript_log.append(f"AI: {ai_text}")
            last_ai_turn_time = time.time()
            await speak_response(ai_text)

        print(f"🎤 Starting AI conversation with {name}")
        speaking_task = asyncio.create_task(
            speak_response(f"Hi {name}, this is Anna. Can you hear me okay?")
        )

 
        async def receive_messages():
            nonlocal stream_sid, call_active
            try:
                while call_active:
                    try:
                        data = await asyncio.wait_for(ws.receive_json(), timeout=0.5)
                        if data.get("event") == "start":
                            stream_sid = data.get("start", {}).get("streamSid")
                        elif data.get("event") == "media":
                            # Forward audio to Deepgram STT
                            media_payload = data.get("media", {}).get("payload", "")
                            if media_payload:
                                audio_bytes = base64.b64decode(media_payload)
                                await dg_stt.send(audio_bytes)
                        elif data.get("event") == "stop":
                            call_active = False
                            break
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"Error receiving message: {e}")
                        break
            except Exception as e:
                print(f"Receive messages error: {e}")

        # Handle Deepgram STT responses
        async def process_stt():
            nonlocal last_user_speech_time
            try:
                while call_active:
                    try:
                        msg = await asyncio.wait_for(dg_stt.recv(), timeout=0.5)
                        if isinstance(msg, str):
                            result = json.loads(msg)
                            if result.get("type") == "Results":
                                transcript = result.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                                if transcript:
                                    transcript_log.append(f"User: {transcript}")
                                    conversation.append({"role": "user", "content": transcript})
                                    last_user_speech_time = time.time()
                                    # Generate AI response
                                    if time.time() - last_ai_turn_time > SILENCE_TIMEOUT:
                                        await generate_and_speak()
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"Error processing STT: {e}")
                        break
            except Exception as e:
                print(f"Process STT error: {e}")


        receive_task = asyncio.create_task(receive_messages())
        stt_task = asyncio.create_task(process_stt())

        while call_active:
            await asyncio.sleep(0.1)
            if time.time() - call_start_time > MAX_CALL_DURATION:
                call_active = False
                break
      
            if time.time() - last_user_speech_time > MAX_INITIAL_SILENCE and last_user_speech_time == call_start_time:
                call_active = False
                break

        receive_task.cancel()
        stt_task.cancel()
        try:
            await receive_task
            await stt_task
        except asyncio.CancelledError:
            pass

    finally:
        try:
            await dg_stt.close()
            await dg_tts.close()
        except:
            pass
        transcript = "\n".join(transcript_log)
        label = await classify_call(transcript)
        print(f"📝 Call ended with {name} ({phone})")
        print(f"📊 Classification: {label}")
        if transcript:
            print(f"💬 Transcript:\n{transcript}")
        save_result(phone, name, transcript, label)
        call_semaphore.release()
        print(f"✅ Call completed and results saved")

# ======================================================
# STARTUP & HEALTH CHECK
# ======================================================

@app.get("/")
async def root():
    """Root endpoint - always works"""
    return {"status": "ok", "service": "voicebot", "message": "API is running"}

@app.get("/health")
async def health():
    """Health check endpoint for Railway"""
    try:
        missing = validate_env()
        return {
            "status": "healthy",
            "app": "running",
            "missing_env_vars": len(missing),
            "warnings": missing if missing else None
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.get("/ping")
async def ping():
    """Simple ping endpoint for testing"""
    return {"pong": True, "timestamp": datetime.utcnow().isoformat()}

# ======================================================
# MAIN ENTRY POINT FOR RAILWAY
# ======================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
