# server.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
import os, datetime, threading, wave
import google.generativeai as genai

app = FastAPI(title="ESP32 Audio Receiver - Gemini STT Server")

UPLOAD_FOLDER = "audio_files"
RAW_FOLDER = "raw_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RAW_FOLDER, exist_ok=True)

CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16000

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise Exception("Set GEMINI_API_KEY env var!")

genai.configure(api_key=GEMINI_KEY)

# Prefer audio-capable model; change if your account supports different name.
STT_MODEL_NAME = "gemini-2.0-flash-lite-preview-02-05"     

TEXT_MODEL_NAME = "gemini-flash-latest"

stt_model = genai.GenerativeModel(STT_MODEL_NAME)
text_model = genai.GenerativeModel(TEXT_MODEL_NAME)

server_status = {
    "running": True,
    "uploads": {},
    "last_recording": None
}

def process_audio_file(raw_path, wav_path):
    try:
        # read raw pcm16 little-endian produced by ESP32 conversion
        with open(raw_path, "rb") as f:
            raw = f.read()

        if len(raw) < SAMPLE_RATE * 1 * 2:  # <1 second (safety)
            print("[ERROR] audio too short:", len(raw))
            return {"success": False, "error": "audio too short"}

        # write WAV header
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw)

        print(f"[OK] WAV saved → {wav_path}")

        # send to Gemini STT (audio-capable model)
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        response = stt_model.generate_content(
            contents=[
                {"mime_type": "audio/wav", "data": audio_bytes},
                "Transcribe this audio into English only. Output only the transcription text."
            ]
        )

        english_text = response.text.strip()
        print("[STT]", english_text)

        # translate to Indonesian
        trans_resp = text_model.generate_content(
            f"Translate the following English text to Indonesian. Output only the translation.\n\n{english_text}"
        )
        ind_text = trans_resp.text.strip()
        print("[ID ]", ind_text)

        result = {"success": True, "english": english_text, "indonesian": ind_text, "file": wav_path}
        return result

    except Exception as e:
        print("[ERROR PROCESS]", repr(e))
        return {"success": False, "error": str(e)}

@app.post("/upload/start")
async def upload_start():
    file_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(RAW_FOLDER, f"{file_id}.raw")
    wav_path = os.path.join(UPLOAD_FOLDER, f"record_{file_id}.wav")
    open(raw_path, "wb").close()
    server_status["uploads"][file_id] = {"raw_path": raw_path, "wav_path": wav_path, "status": "uploading", "result": None}
    return {"id": file_id}

@app.post("/upload/chunk/{file_id}")
async def upload_chunk(file_id: str, request: Request):
    if file_id not in server_status["uploads"]:
        raise HTTPException(404, "file_id not found")
    info = server_status["uploads"][file_id]
    raw_path = info["raw_path"]
    try:
        chunk = await request.body()
        if not chunk:
            raise HTTPException(422, "empty chunk")
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
        res = process_audio_file(info["raw_path"], info["wav_path"])
        info["result"] = res
        info["status"] = "done"
        server_status["last_recording"] = res
    threading.Thread(target=job, daemon=True).start()
    return {"ok": True, "message": "processing started"}

@app.get("/last-recording")
async def last_recording():
    if server_status["last_recording"]:
        return server_status["last_recording"]
    return {"message": "No recordings yet"}

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "file not found")
    return FileResponse(file_path, filename=filename, media_type="audio/wav")

@app.get("/status")
async def status():
    return {"uploads": server_status["uploads"]}

@app.get("/")
async def root():
    return {"message": "Gemini STT Server running"}
