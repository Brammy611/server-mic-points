from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
import os, datetime, threading, wave
import google.generativeai as genai

app = FastAPI(title="ESP32 Audio Receiver - Gemini STT Server")

# ======================================
# 📂 FOLDERS
# ======================================
UPLOAD_FOLDER = "audio_files"
RAW_FOLDER = "raw_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RAW_FOLDER, exist_ok=True)

CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16000

# ======================================
# 🔑 GEMINI API KEY
# ======================================
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise Exception("GEMINI_API_KEY environment variable is NOT set!")

genai.configure(api_key=api_key)

# Model untuk Speech-to-Text
stt_model = genai.GenerativeModel("gemini-2.0-flash-lite-preview-02-05")

# Model untuk translate
text_model = genai.GenerativeModel("gemini-flash-latest")

# ======================================
# 🧠 SERVER STATE
# ======================================
server_status = {
    "running": True,
    "uploads": {},
    "last_recording": None
}

# ======================================
# 🔊 AUDIO PROCESS
# ======================================
def process_audio_file(raw_path, wav_path):
    try:
        # RAW → WAV
        with open(raw_path, "rb") as f:
            raw_data = f.read()

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw_data)

        print(f"[OK] WAV saved → {wav_path}")

        # --- STT with Gemini ---
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
            response = stt_model.generate_content(
                contents=[
                    {"mime_type": "audio/wav", "data": audio_bytes},
                    "Transcribe the audio into English only."
                ]
            )

        english_text = response.text
        print("[STT]", english_text)

        # --- TRANSLATE ---
        translation = text_model.generate_content(
            f"Translate this to Indonesian:\n{english_text}"
        ).text

        print("[ID ]", translation)

        result = {
            "success": True,
            "english": english_text,
            "indonesian": translation,
            "file": wav_path
        }

        return result

    except Exception as e:
        print("[ERROR]", str(e))
        return {"success": False, "error": str(e)}


# ======================================
# 📌 API ENDPOINTS
# ======================================

@app.post("/upload/start")
async def upload_start():
    file_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(RAW_FOLDER, f"{file_id}.raw")
    wav_path = os.path.join(UPLOAD_FOLDER, f"record_{file_id}.wav")

    open(raw_path, "wb").close()

    server_status["uploads"][file_id] = {
        "raw_path": raw_path,
        "wav_path": wav_path,
        "status": "uploading",
        "result": None
    }

    return {"id": file_id}


@app.post("/upload/chunk/{file_id}")
async def upload_chunk(file_id: str, request: Request):
    if file_id not in server_status["uploads"]:
        raise HTTPException(404, "file_id not found")

    info = server_status["uploads"][file_id]
    raw_path = info["raw_path"]

    try:
        chunk = await request.body()
        with open(raw_path, "ab") as f:
            f.write(chunk)

        return {"ok": True, "received_bytes": len(chunk)}

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/upload/finish/{file_id}")
async def upload_finish(file_id: str):
    if file_id not in server_status["uploads"]:
        raise HTTPException(404, "file_id not found")

    info = server_status["uploads"][file_id]

    if info["status"] != "uploading":
        return {"ok": False, "message": "already processed"}

    info["status"] = "processing"

    def job():
        result = process_audio_file(info["raw_path"], info["wav_path"])
        info["result"] = result
        info["status"] = "done"
        server_status["last_recording"] = result

    threading.Thread(target=job, daemon=True).start()

    return {"ok": True, "message": "processing started"}


@app.get("/last-recording")
async def last_recording():
    if server_status["last_recording"]:
        return server_status["last_recording"]
    return {"message": "no recordings yet"}


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"{UPLOAD_FOLDER}/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(404, "file not found")
    return FileResponse(file_path, filename=filename)


@app.get("/status")
async def status():
    return {"uploads": server_status["uploads"]}


@app.get("/")
async def home():
    return {"message": "Gemini STT Server Running"}


