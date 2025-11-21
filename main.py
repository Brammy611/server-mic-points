# server.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
import os, datetime, threading, wave
import speech_recognition as sr
from googletrans import Translator

app = FastAPI(title="ESP32 Audio Receiver - HTTP Upload Server")

UPLOAD_FOLDER = "audio_files"
RAW_FOLDER = "raw_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RAW_FOLDER, exist_ok=True)

# audio params (ESP32 PCM)
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (16-bit)
SAMPLE_RATE = 16000

# status
server_status = {
    "running": True,
    "uploads": {},  # id -> {"raw_path":..., "wav_path":..., "status": "uploading/processing/done", "result": None}
    "last_recording": None
}

def process_audio_file(raw_path, wav_path):
    """Read raw PCM file, convert to WAV, transcribe and translate."""
    try:
        # Read raw bytes
        with open(raw_path, "rb") as f:
            raw = f.read()

        # Write proper WAV file
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw)

        print(f"Saved WAV: {wav_path}")

        # Speech to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as src:
            audio = recognizer.record(src)

        text_en = recognizer.recognize_google(audio, language="en-US")
        translator = Translator()
        translated = translator.translate(text_en, src='en', dest='id').text

        result = {"success": True, "english": text_en, "indonesian": translated, "file": wav_path}
        print("Transcription:", text_en)
        print("Translation:", translated)

        return result

    except sr.UnknownValueError:
        return {"success": False, "error": "Speech not recognized"}
    except sr.RequestError as e:
        return {"success": False, "error": f"Speech API error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/upload/start")
async def upload_start():
    """Start new upload session, return an id."""
    file_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(RAW_FOLDER, f"{file_id}.raw")
    wav_path = os.path.join(UPLOAD_FOLDER, f"record_{file_id}.wav")
    # create empty raw file
    open(raw_path, "wb").close()
    server_status["uploads"][file_id] = {"raw_path": raw_path, "wav_path": wav_path, "status": "uploading", "result": None}
    return {"id": file_id}

@app.post("/upload/chunk/{file_id}")
async def upload_chunk(file_id: str, request: Request):
    if file_id not in server_status["uploads"]:
        raise HTTPException(status_code=404, detail="file_id not found")

    info = server_status["uploads"][file_id]
    raw_path = info["raw_path"]

    try:
        body = await request.body()  # raw bytes from ESP32
        with open(raw_path, "ab") as f:
            f.write(body)
        return {"ok": True, "received_bytes": len(body)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/finish/{file_id}")
async def upload_finish(file_id: str):
    """Finish upload: convert & process in background thread."""
    if file_id not in server_status["uploads"]:
        raise HTTPException(status_code=404, detail="file_id not found")
    info = server_status["uploads"][file_id]
    if info["status"] != "uploading":
        return {"ok": False, "message": "Already finished or processing"}

    info["status"] = "processing"

    def job():
        res = process_audio_file(info["raw_path"], info["wav_path"])
        info["result"] = res
        info["status"] = "done"
        server_status["last_recording"] = res

    t = threading.Thread(target=job, daemon=True)
    t.start()

    return {"ok": True, "message": "processing started"}

@app.get("/last-recording")
async def last_recording():
    if server_status["last_recording"]:
        return server_status["last_recording"]
    return {"message": "No recordings yet"}

@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = f"audio_files/{filename}"
    return FileResponse(path=file_path, filename=filename, media_type="audio/wav")
    
@app.get("/status")
async def status():
    return {"uploads": server_status["uploads"]}

# Run with uvicorn on PORT env
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


