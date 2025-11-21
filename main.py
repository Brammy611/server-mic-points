from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI
import os, datetime, threading, wave

# =======================
# FASTAPI APP
# =======================
app = FastAPI(title="ESP32 Audio Receiver - Whisper Cloud")

# =======================
# DIRECTORIES
# =======================
UPLOAD_FOLDER = "audio_files"
RAW_FOLDER = "raw_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RAW_FOLDER, exist_ok=True)

# =======================
# AUDIO CONFIG (ESP32 PCM)
# =======================
CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16000

# =======================
# OPENAI CLIENT
# =======================
client = OpenAI()

# =======================
# SERVER STATUS
# =======================
server_status = {
    "running": True,
    "uploads": {},
    "last_recording": None
}

# =======================
# TRANSCRIBE + TRANSLATE (OpenAI Online)
# =======================
def transcribe_and_translate(wav_path):
    # --- TRANSCRIBE ---
    with open(wav_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-tts",
            file=f
        )
    english_text = transcription.text

    # --- TRANSLATE ---
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Translate the following English text to Indonesian."},
            {"role": "user", "content": english_text},
        ]
    )
    ind_text = completion.choices[0].message["content"]

    return english_text, ind_text

# =======================
# PCM → WAV + STT thread
# =======================
def process_audio_file(raw_path, wav_path):
    try:
        # Convert RAW → WAV
        with open(raw_path, "rb") as f:
            pcm = f.read()

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)

        print(f"[OK] WAV saved → {wav_path}")

        # STT + Translate
        eng, indo = transcribe_and_translate(wav_path)

        result = {
            "success": True,
            "english": eng,
            "indonesian": indo,
            "file": wav_path
        }

        print("[TRANSCRIPTION]", eng)
        print("[TRANSLATION]", indo)

        return result

    except Exception as e:
        print("[ERROR PROCESSING]", e)
        return {"success": False, "error": str(e)}

# =======================
# START UPLOAD
# =======================
@app.post("/upload/start")
async def upload_start():
    file_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = f"{RAW_FOLDER}/{file_id}.raw"
    wav_path = f"{UPLOAD_FOLDER}/record_{file_id}.wav"

    open(raw_path, "wb").close()

    server_status["uploads"][file_id] = {
        "raw_path": raw_path,
        "wav_path": wav_path,
        "status": "uploading",
        "result": None
    }

    return {"id": file_id}

# =======================
# RECEIVE CHUNK
# =======================
@app.post("/upload/chunk/{file_id}")
async def upload_chunk(file_id: str, request: Request):
    if file_id not in server_status["uploads"]:
        raise HTTPException(status_code=404, detail="file_id not found")

    info = server_status["uploads"][file_id]
    raw_path = info["raw_path"]

    try:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=422, detail="empty chunk")

        with open(raw_path, "ab") as f:
            f.write(body)

        return {"received": len(body)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chunk error: {e}")

# =======================
# FINISH UPLOAD → PROCESS THREAD
# =======================
@app.post("/upload/finish/{file_id}")
async def upload_finish(file_id: str):
    if file_id not in server_status["uploads"]:
        raise HTTPException(status_code=404, detail="file_id not found")

    info = server_status["uploads"][file_id]
    if info["status"] != "uploading":
        return {"ok": False, "message": "already processing or done"}

    info["status"] = "processing"

    def job():
        res = process_audio_file(info["raw_path"], info["wav_path"])
        info["result"] = res
        info["status"] = "done"
        server_status["last_recording"] = res

    threading.Thread(target=job, daemon=True).start()

    return {"ok": True, "message": "processing started"}

# =======================
# GET LAST RESULT
# =======================
@app.get("/last-recording")
async def last_recording():
    if server_status["last_recording"]:
        return server_status["last_recording"]
    return {"message": "No recordings yet"}

# =======================
# DOWNLOAD FILE
# =======================
@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = f"{UPLOAD_FOLDER}/{filename}"
    return FileResponse(path=file_path, filename=filename, media_type="audio/wav")

# =======================
# SERVER STATUS
# =======================
@app.get("/status")
async def status():
    return server_status

# =======================
# RUN LOCAL
# =======================
if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
