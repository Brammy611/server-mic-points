from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
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

# Load Gemini API Key (pasang di Railway ENV)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

server_status = {
    "running": True,
    "uploads": {},
    "last_recording": None
}


def process_audio_file(raw_path, wav_path):
    try:
        # Convert RAW → WAV
        with open(raw_path, "rb") as f:
            raw = f.read()

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw)

        print(f"[OK] WAV saved → {wav_path}")

        # --- GEMINI STT ---
        with open(wav_path, "rb") as audio_file:
            response = model.generate_content(
                contents=[
                    {"mime_type": "audio/wav", "data": audio_file.read()},
                    {"text": "Transcribe this audio into English."}
                ]
            )

        english_text = response.text

        # --- GEMINI Translation ---
        translation = model.generate_content(
            f"Translate this text to Indonesian: {english_text}"
        ).text

        result = {
            "success": True,
            "english": english_text,
            "indonesian": translation,
            "file": wav_path
        }

        print("[STT]", english_text)
        print("[ID ]", translation)

        return result

    except Exception as e:
        print("[ERROR]", str(e))
        return {"success": False, "error": str(e)}


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
        body = await request.body()
        with open(raw_path, "ab") as f:
            f.write(body)
        return {"ok": True, "received_bytes": len(body)}

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/upload/finish/{file_id}")
async def upload_finish(file_id: str):
    if file_id not in server_status["uploads"]:
        raise HTTPException(404, "file_id not found")

    info = server_status["uploads"][file_id]
    if info["status"] != "uploading":
        return {"ok": False, "message": "Already processed"}

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
def download_file(filename: str):
    file_path = f"audio_files/{filename}"
    return FileResponse(file_path, filename=filename)


@app.get("/status")
async def status():
    return {"uploads": server_status["uploads"]}
