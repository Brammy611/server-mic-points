from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import wave
import datetime
import speech_recognition as sr
from googletrans import Translator
import io
import os

app = FastAPI(title="ESP32 Audio Receiver API")

# Konfigurasi audio
CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16000

# Folder untuk menyimpan file
UPLOAD_FOLDER = "audio_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "ESP32 Audio Receiver API is running",
        "endpoints": {
            "/upload-audio": "POST - Upload audio data from ESP32",
            "/health": "GET - Check server health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    Endpoint untuk menerima file audio dari ESP32
    ESP32 akan mengirim raw audio bytes sebagai file
    """
    try:
        # Baca data audio dari upload
        audio_data = await file.read()
        
        # Generate timestamp untuk nama file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(UPLOAD_FOLDER, f"record_{timestamp}.wav")
        
        # Simpan sebagai file WAV
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)
        
        print(f"✅ Audio tersimpan sebagai {output_file}")
        
        # ============= SPEECH TO TEXT =============
        print("🧠 Mengubah suara bahasa Inggris menjadi teks...")
        recognizer = sr.Recognizer()
        
        try:
            with sr.AudioFile(output_file) as source:
                audio = recognizer.record(source)
            
            # Speech-to-text dari bahasa Inggris 🇬🇧
            text_en = recognizer.recognize_google(audio, language="en-US")
            print("📄 Hasil transkripsi (English):")
            print(text_en)
            
            # Translasi ke Bahasa Indonesia 🇮🇩
            translator = Translator()
            translated = translator.translate(text_en, src='en', dest='id').text
            
            print("\n🇮🇩 Hasil terjemahan:")
            print(translated)
            
            # Simpan dua versi (EN & ID)
            txt_file = output_file.replace('.wav', '_translated.txt')
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("=== English ===\n")
                f.write(text_en + "\n\n")
                f.write("=== Indonesian ===\n")
                f.write(translated)
            
            print("💾 File teks disimpan!")
            
            return JSONResponse(content={
                "status": "success",
                "message": "Audio processed successfully",
                "filename": output_file,
                "transcription": {
                    "english": text_en,
                    "indonesian": translated
                },
                "timestamp": timestamp
            })
            
        except sr.UnknownValueError:
            print("⚠️ Suara tidak terdeteksi atau tidak bisa dikenali.")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Suara tidak terdeteksi atau tidak bisa dikenali",
                    "filename": output_file
                }
            )
        except sr.RequestError as e:
            print(f"❌ Error koneksi ke Google Speech API: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": f"Error koneksi ke Google Speech API: {str(e)}",
                    "filename": output_file
                }
            )
        except Exception as e:
            print(f"⚠️ Error lain: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Error processing audio: {str(e)}",
                    "filename": output_file
                }
            )
            
    except Exception as e:
        print(f"❌ Error saat menerima file: {e}")
        raise HTTPException(status_code=500, detail=f"Error receiving file: {str(e)}")

# Untuk menjalankan di development
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
