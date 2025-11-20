from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import wave
import datetime
import speech_recognition as sr
from googletrans import Translator
import os
import asyncio

app = FastAPI(title="ESP32 Audio Receiver API - Chunked Upload")

# Konfigurasi audio
CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16000

# Folder untuk menyimpan file
UPLOAD_FOLDER = "audio_files"
CHUNKS_FOLDER = "audio_chunks"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHUNKS_FOLDER, exist_ok=True)

# Dictionary untuk track sessions
sessions = {}

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "ESP32 Audio Receiver API - Chunked Upload",
        "endpoints": {
            "/upload-audio": "POST - Upload complete audio (legacy)",
            "/upload-audio-chunk": "POST - Upload audio chunk by chunk",
            "/health": "GET - Check server health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.post("/upload-audio-chunk")
async def upload_audio_chunk(
    session_id: str = Form(...),
    chunk_number: int = Form(...),
    is_final: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Endpoint untuk menerima audio chunks dari ESP32
    ESP32 akan mengirim audio dalam potongan-potongan kecil
    """
    try:
        is_final_chunk = is_final.lower() == "true"
        
        print(f"\n{'='*60}")
        print(f"📥 Receiving chunk #{chunk_number} for session {session_id[:16]}...")
        print(f"📋 Is final: {is_final_chunk}")
        
        # Baca chunk data
        chunk_data = await file.read()
        print(f"📊 Received {len(chunk_data)} bytes")
        
        # Initialize session jika belum ada
        if session_id not in sessions:
            sessions[session_id] = {
                'chunks': {},
                'start_time': datetime.datetime.now(),
                'total_chunks': 0
            }
            print(f"🆕 New session created: {session_id[:16]}...")
        
        # Simpan chunk
        sessions[session_id]['chunks'][chunk_number] = chunk_data
        sessions[session_id]['total_chunks'] = max(
            sessions[session_id]['total_chunks'], 
            chunk_number
        )
        
        print(f"✅ Chunk #{chunk_number} saved")
        print(f"📦 Total chunks for this session: {len(sessions[session_id]['chunks'])}")
        
        # Jika ini chunk terakhir, proses semua chunks
        if is_final_chunk:
            print(f"\n🔄 Processing final audio for session {session_id[:16]}...")
            return await process_complete_audio(session_id)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Chunk {chunk_number} received",
            "session_id": session_id,
            "chunks_received": len(sessions[session_id]['chunks'])
        })
        
    except Exception as e:
        print(f"❌ Error receiving chunk: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

async def process_complete_audio(session_id: str):
    """Gabungkan semua chunks dan proses audio"""
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        chunks = session['chunks']
        
        print(f"\n{'='*60}")
        print(f"🔧 PROCESSING COMPLETE AUDIO")
        print(f"{'='*60}")
        print(f"🆔 Session: {session_id[:16]}...")
        print(f"📦 Total chunks: {len(chunks)}")
        
        # Gabungkan semua chunks sesuai urutan
        complete_audio = bytearray()
        for i in sorted(chunks.keys()):
            complete_audio.extend(chunks[i])
            print(f"➕ Added chunk #{i} ({len(chunks[i])} bytes)")
        
        total_size = len(complete_audio)
        duration = total_size / (SAMPLE_RATE * SAMPLE_WIDTH)
        
        print(f"\n📊 Complete audio stats:")
        print(f"   Total size: {total_size:,} bytes ({total_size//1024} KB)")
        print(f"   Duration: {duration:.2f} seconds")
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(UPLOAD_FOLDER, f"record_{timestamp}.wav")
        
        # Simpan sebagai WAV file
        print(f"\n💾 Saving to: {output_file}")
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(complete_audio)
        
        print(f"✅ WAV file saved")
        
        # ============= SPEECH TO TEXT =============
        print(f"\n🧠 Converting speech to text...")
        recognizer = sr.Recognizer()
        
        try:
            with sr.AudioFile(output_file) as source:
                audio = recognizer.record(source)
            
            print("📡 Calling Google Speech API...")
            text_en = recognizer.recognize_google(audio, language="en-US")
            print(f"📄 Transcription (EN): {text_en}")
            
            # Translasi ke Bahasa Indonesia
            print("🌐 Translating to Indonesian...")
            translator = Translator()
            translated = translator.translate(text_en, src='en', dest='id').text
            print(f"🇮🇩 Translation (ID): {translated}")
            
            # Simpan hasil
            txt_file = output_file.replace('.wav', '_translated.txt')
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"Chunks: {len(chunks)}\n\n")
                f.write("=== English ===\n")
                f.write(text_en + "\n\n")
                f.write("=== Indonesian ===\n")
                f.write(translated)
            
            print("💾 Transcription saved!")
            
            # Cleanup session
            del sessions[session_id]
            print(f"🗑️ Session {session_id[:16]}... cleaned up")
            
            print(f"\n{'='*60}")
            print(f"✅ PROCESSING COMPLETE!")
            print(f"{'='*60}\n")
            
            return JSONResponse(content={
                "status": "success",
                "message": "Audio processed successfully",
                "session_id": session_id,
                "filename": output_file,
                "duration": duration,
                "chunks_processed": len(chunks),
                "transcription": {
                    "english": text_en,
                    "indonesian": translated
                },
                "timestamp": timestamp
            })
            
        except sr.UnknownValueError:
            print("⚠️ Speech not recognized")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Suara tidak terdeteksi atau tidak bisa dikenali",
                    "filename": output_file,
                    "duration": duration
                }
            )
        except sr.RequestError as e:
            print(f"❌ Google Speech API error: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": f"Error koneksi ke Google Speech API: {str(e)}",
                    "filename": output_file
                }
            )
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Error processing audio: {str(e)}",
                    "filename": output_file
                }
            )
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    Legacy endpoint - untuk backward compatibility
    """
    try:
        print(f"\n📥 Receiving complete audio file...")
        audio_data = await file.read()
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(UPLOAD_FOLDER, f"record_{timestamp}.wav")
        
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)
        
        print(f"✅ Audio saved: {output_file}")
        
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(output_file) as source:
            audio = recognizer.record(source)
        
        text_en = recognizer.recognize_google(audio, language="en-US")
        translator = Translator()
        translated = translator.translate(text_en, src='en', dest='id').text
        
        return JSONResponse(content={
            "status": "success",
            "message": "Audio processed successfully",
            "filename": output_file,
            "transcription": {
                "english": text_en,
                "indonesian": translated
            }
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cleanup old sessions periodically
@app.on_event("startup")
async def startup_event():
    """Background task untuk cleanup old sessions"""
    async def cleanup_old_sessions():
        while True:
            await asyncio.sleep(600)  # Every 10 minutes
            now = datetime.datetime.now()
            to_delete = []
            
            for session_id, session in sessions.items():
                age = (now - session['start_time']).total_seconds()
                if age > 1800:  # 30 minutes
                    to_delete.append(session_id)
            
            for session_id in to_delete:
                print(f"🗑️ Cleaning up old session: {session_id[:16]}...")
                del sessions[session_id]
    
    asyncio.create_task(cleanup_old_sessions())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
