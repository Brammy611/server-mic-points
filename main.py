from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os, datetime, threading, wave
import google.generativeai as genai
from pymongo import MongoClient
from bson import ObjectId

app = FastAPI(title="ESP32 Audio Receiver - Gemini STT Server")

# CORS untuk akses dari berbagai origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "audio_files"
RAW_FOLDER = "raw_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RAW_FOLDER, exist_ok=True)

CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16000

# Konfigurasi API Keys
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise Exception("Set GEMINI_API_KEY env var!")

genai.configure(api_key=GEMINI_KEY)

# MongoDB Atlas Configuration
MONGODB_URI = os.environ.get("MONGODB_URI")
if not MONGODB_URI:
    raise Exception("Set MONGODB_URI env var! Format: mongodb+srv://username:password@cluster.mongodb.net/")

# Connect to MongoDB
try:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client["audio_transcription"]  # Database name
    recordings_collection = db["recordings"]  # Collection name
    print("[OK] Connected to MongoDB Atlas")
except Exception as e:
    print(f"[ERROR] MongoDB connection failed: {e}")
    raise

# Gemini Models
STT_MODEL_NAME = "gemini-2.0-flash-lite-preview-02-05"     
TEXT_MODEL_NAME = "gemini-flash-latest"

stt_model = genai.GenerativeModel(STT_MODEL_NAME)
text_model = genai.GenerativeModel(TEXT_MODEL_NAME)

server_status = {
    "running": True,
    "uploads": {},
    "last_recording": None
}

def save_to_mongodb(result_data):
    """Save transcription result to MongoDB"""
    try:
        doc = {
            "timestamp": datetime.datetime.utcnow(),
            "english_text": result_data.get("english", ""),
            "indonesian_text": result_data.get("indonesian", ""),
            "file_path": result_data.get("file", ""),
            "success": result_data.get("success", False),
            "error": result_data.get("error", None),
            "audio_duration_seconds": result_data.get("duration", 0),
            "metadata": {
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS
            }
        }
        insert_result = recordings_collection.insert_one(doc)
        print(f"[MongoDB] Saved with ID: {insert_result.inserted_id}")
        return str(insert_result.inserted_id)
    except Exception as e:
        print(f"[ERROR] MongoDB save failed: {e}")
        return None

def process_audio_file(raw_path, wav_path):
    try:
        # read raw pcm16 little-endian produced by ESP32 conversion
        with open(raw_path, "rb") as f:
            raw = f.read()

        # Minimal 1.5 detik audio (16000 Hz * 2 bytes * 1.5 seconds)
        min_bytes = int(SAMPLE_RATE * SAMPLE_WIDTH * 1.5)
        if len(raw) < min_bytes:
            print(f"[ERROR] audio too short: {len(raw)} bytes ({len(raw)/(SAMPLE_RATE*SAMPLE_WIDTH):.2f}s), need at least {min_bytes} bytes (1.5s)")
            return {"success": False, "error": f"audio too short: {len(raw)/(SAMPLE_RATE*SAMPLE_WIDTH):.2f}s, need at least 1.5s"}

        # Calculate duration
        duration = len(raw) / (SAMPLE_RATE * SAMPLE_WIDTH)

        # write WAV header
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw)

        print(f"[OK] WAV saved → {wav_path} (duration: {duration:.2f}s)")

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

        result = {
            "success": True, 
            "english": english_text, 
            "indonesian": ind_text, 
            "file": wav_path,
            "duration": duration
        }
        
        # Save to MongoDB
        mongo_id = save_to_mongodb(result)
        if mongo_id:
            result["mongo_id"] = mongo_id
        
        return result

    except Exception as e:
        print("[ERROR PROCESS]", repr(e))
        error_result = {"success": False, "error": str(e)}
        save_to_mongodb(error_result)
        return error_result

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
        if not chunk:
            raise HTTPException(422, "empty chunk")
        with open(raw_path, "ab") as f:
            f.write(chunk)
        return {"ok": True, "received_bytes": len(chunk)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/upload/finish/{file_id}")
async def upload_finish(file_id: str):
    print(f"[FINISH] Received finish request for file_id: {file_id}")
    if file_id not in server_status["uploads"]:
        print(f"[FINISH ERROR] file_id not found: {file_id}")
        raise HTTPException(404, "file_id not found")
    info = server_status["uploads"][file_id]
    if info["status"] != "uploading":
        print(f"[FINISH] Already processed, status: {info['status']}")
        return {"ok": False, "message": "already processed"}
    info["status"] = "processing"
    print(f"[FINISH] Starting processing thread for {file_id}")
    def job():
        print(f"[PROCESSING] Starting audio processing for {file_id}")
        res = process_audio_file(info["raw_path"], info["wav_path"])
        info["result"] = res
        info["status"] = "done"
        server_status["last_recording"] = res
        print(f"[PROCESSING] Finished processing {file_id}, result: {res.get('success', False)}")
    threading.Thread(target=job, daemon=True).start()
    print(f"[FINISH] Processing thread started for {file_id}")
    return {"ok": True, "message": "processing started"}

@app.get("/last-recording")
async def last_recording():
    if server_status["last_recording"]:
        return server_status["last_recording"]
    return {"message": "No recordings yet"}

@app.get("/recordings")
async def get_recordings(limit: int = 10, skip: int = 0):
    """Get recordings from MongoDB"""
    try:
        recordings = list(
            recordings_collection
            .find()
            .sort("timestamp", -1)
            .skip(skip)
            .limit(limit)
        )
        # Convert ObjectId to string
        for rec in recordings:
            rec["_id"] = str(rec["_id"])
        return {"recordings": recordings, "count": len(recordings)}
    except Exception as e:
        raise HTTPException(500, f"Database error: {str(e)}")

@app.get("/recording/{recording_id}")
async def get_recording(recording_id: str):
    """Get specific recording by ID"""
    try:
        recording = recordings_collection.find_one({"_id": ObjectId(recording_id)})
        if not recording:
            raise HTTPException(404, "Recording not found")
        recording["_id"] = str(recording["_id"])
        return recording
    except Exception as e:
        raise HTTPException(500, f"Database error: {str(e)}")

@app.delete("/recording/{recording_id}")
async def delete_recording(recording_id: str):
    """Delete recording from MongoDB"""
    try:
        result = recordings_collection.delete_one({"_id": ObjectId(recording_id)})
        if result.deleted_count == 0:
            raise HTTPException(404, "Recording not found")
        return {"ok": True, "message": "Recording deleted"}
    except Exception as e:
        raise HTTPException(500, f"Database error: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "file not found")
    return FileResponse(file_path, filename=filename, media_type="audio/wav")

@app.get("/status")
async def status():
    return {
        "uploads": server_status["uploads"],
        "mongodb_connected": mongo_client is not None
    }

@app.get("/")
async def root():
    return {
        "message": "Gemini STT Server running",
        "mongodb": "connected",
        "endpoints": {
            "upload": "/upload/start, /upload/chunk, /upload/finish",
            "recordings": "/recordings, /recording/{id}",
            "last": "/last-recording"
        }
    }

@app.on_event("shutdown")
async def shutdown():
    """Close MongoDB connection on shutdown"""
    if mongo_client:
        mongo_client.close()
        print("[OK] MongoDB connection closed")
