import os
import uuid
import warnings
import json
import sys
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from voice_logic import (
    openai_client, get_whisper_model, transcribe_audio, load_chat_history, save_chat_message,
    clean_audio_gentle, generate_ai_response, text_to_speech, TEMP_DIR
)

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')

# INIT
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend HTML
@app.get("/", response_class=HTMLResponse)
def serve_root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/history/{session_id}")
def get_chat_history(session_id: str):
    """Get chat history for a session"""
    history = load_chat_history(session_id)
    return {"session_id": session_id, "history": history}

@app.post("/new-session")
def create_new_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

# API
@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...), language: str = Form(None), session_id: str = Form(None)):
    try:
        # Save uploaded audio
        input_audio_path = f"{TEMP_DIR}/{uuid.uuid4()}.wav"
        with open(input_audio_path, "wb") as f:
            f.write(await audio.read())

        # Apply gentle preprocessing for all languages
        cleaned_audio_path = f"{TEMP_DIR}/{uuid.uuid4()}_clean.wav"
        cleaning_success = clean_audio_gentle(input_audio_path, cleaned_audio_path)
        audio_to_transcribe = cleaned_audio_path if cleaning_success else input_audio_path

        print(f"Selected language: {language}")

        # Transcribe
        transcribe_lang = language or 'en'
        print(f"Transcribing in language: {transcribe_lang}")
        user_text = transcribe_audio(audio_to_transcribe, transcribe_lang)

        detected_language = transcribe_lang
        language_probability = 1.0
        
        print("="*50)
        print(f"TRANSCRIPTION RESULT:")
        print(f"  Selected: {language}")
        print(f"  Text: {user_text}")
        print("="*50)

        # Generate bot reply
        bot_reply = generate_ai_response(user_text, session_id, transcribe_lang)
        
        # Save history
        if session_id:
            save_chat_message(session_id, user_text, bot_reply, transcribe_lang)

        # TTS
        output_audio_path = f"{TEMP_DIR}/{uuid.uuid4()}.mp3"
        text_to_speech(bot_reply, transcribe_lang, output_audio_path)

        return {
            "session_id": session_id,
            "detected_language": detected_language,
            "language_confidence": f"{language_probability:.2%}",
            "user_text": user_text,
            "bot_text": bot_reply,
            "audio_url": f"/audio/{os.path.basename(output_audio_path)}"
        }
        
    except Exception as e:
        print(f"ERROR in voice_chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Failed to process request: {str(e)}"
        }


@app.get("/audio/{filename}")
def get_audio(filename: str):
    return FileResponse(f"{TEMP_DIR}/{filename}", media_type="audio/mpeg")