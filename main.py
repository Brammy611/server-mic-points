from fastapi import FastAPI
import asyncio
import socket
import wave
import datetime
import speech_recognition as sr
from googletrans import Translator
import os
import threading

app = FastAPI(title="ESP32 Audio Receiver - Socket Server")

# Konfigurasi
HOST = '0.0.0.0'
SOCKET_PORT = 5000
CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16000
UPLOAD_FOLDER = "audio_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Status server
server_status = {
    "running": False,
    "connected": False,
    "recording": False,
    "last_recording": None
}

def process_audio(frames, output_file):
    """Process audio: save, transcribe, translate"""
    try:
        # Simpan sebagai WAV
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(frames)
        
        print(f"✅ Audio tersimpan sebagai {output_file}")
        
        # Speech to text
        print("🧠 Mengubah suara bahasa Inggris menjadi teks...")
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(output_file) as source:
            audio = recognizer.record(source)
        
        try:
            text_en = recognizer.recognize_google(audio, language="en-US")
            print("📄 Hasil transkripsi (English):")
            print(text_en)
            
            translator = Translator()
            translated = translator.translate(text_en, src='en', dest='id').text
            
            print("\n🇮🇩 Hasil terjemahan:")
            print(translated)
            
            # Simpan hasil
            txt_file = output_file.replace('.wav', '_translated.txt')
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("=== English ===\n")
                f.write(text_en + "\n\n")
                f.write("=== Indonesian ===\n")
                f.write(translated)
            
            print("💾 File teks disimpan!")
            
            return {
                "success": True,
                "english": text_en,
                "indonesian": translated,
                "file": output_file
            }
            
        except sr.UnknownValueError:
            print("⚠️ Suara tidak terdeteksi atau tidak bisa dikenali.")
            return {"success": False, "error": "Speech not recognized"}
        except sr.RequestError as e:
            print(f"❌ Error koneksi ke Google Speech API: {e}")
            return {"success": False, "error": str(e)}
            
    except Exception as e:
        print(f"⚠️ Error processing: {e}")
        return {"success": False, "error": str(e)}

def socket_server_thread():
    """TCP Socket server running in background thread"""
    global server_status
    
    print("\n" + "="*60)
    print("🎙 Starting Socket Server...")
    print("="*60)
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, SOCKET_PORT))
    server.listen(1)
    server.settimeout(1.0)  # Timeout untuk bisa check shutdown
    
    server_status["running"] = True
    print(f"✅ Socket server listening on {HOST}:{SOCKET_PORT}")
    
    while server_status["running"]:
        try:
            print("\n🎙 Menunggu koneksi ESP32...")
            conn, addr = server.accept()
            
            print(f"✅ Terhubung dari {addr}")
            server_status["connected"] = True
            server_status["recording"] = True
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(UPLOAD_FOLDER, f"record_{timestamp}.wav")
            
            frames = bytearray()
            
            try:
                while True:
                    data = conn.recv(512)
                    if not data:
                        break
                    frames.extend(data)
                    
                    # Progress update setiap 32KB
                    if len(frames) % 32000 == 0:
                        print(f"📊 Received: {len(frames)//1024} KB")
                        
            except Exception as e:
                print(f"⚠️ Connection error: {e}")
            finally:
                conn.close()
                server_status["connected"] = False
                server_status["recording"] = False
                
                print(f"\n📊 Total received: {len(frames)} bytes ({len(frames)//1024} KB)")
                
                # Process audio
                if len(frames) > 0:
                    result = process_audio(frames, output_file)
                    server_status["last_recording"] = result
                else:
                    print("⚠️ No data received")
                    
        except socket.timeout:
            continue
        except Exception as e:
            print(f"❌ Server error: {e}")
            break
    
    server.close()
    print("🛑 Socket server stopped")

# Start socket server in background
socket_thread = threading.Thread(target=socket_server_thread, daemon=True)
socket_thread.start()

# FastAPI endpoints
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "ESP32 Audio Receiver - Socket Server",
        "socket_server": {
            "host": HOST,
            "port": SOCKET_PORT,
            "running": server_status["running"],
            "connected": server_status["connected"],
            "recording": server_status["recording"]
        },
        "endpoints": {
            "/health": "GET - Check server health",
            "/status": "GET - Get socket server status",
            "/last-recording": "GET - Get last recording result"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "socket_server_running": server_status["running"],
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/status")
async def status():
    return {
        "socket_server": server_status,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/last-recording")
async def last_recording():
    if server_status["last_recording"]:
        return server_status["last_recording"]
    else:
        return {"message": "No recordings yet"}

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n🛑 Shutting down socket server...")
    server_status["running"] = False
    socket_thread.join(timeout=5)

if __name__ == "__main__":
    import uvicorn
    # Get port from environment or use 8000
    http_port = int(os.environ.get("PORT", 8000))
    
    print("\n" + "="*60)
    print("🚀 Starting Hybrid Server")
    print("="*60)
    print(f"📡 HTTP API: http://0.0.0.0:{http_port}")
    print(f"🔌 Socket Server: tcp://0.0.0.0:{SOCKET_PORT}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=http_port)
