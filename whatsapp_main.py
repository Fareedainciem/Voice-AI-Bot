import os
import uuid
import requests
import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import Response, FileResponse
from twilio.twiml.messaging_response import MessagingResponse
from voice_logic import (
    get_whisper_model, transcribe_audio, generate_ai_response, text_to_speech, 
    clean_audio_gentle, save_chat_message, TEMP_DIR,
    get_session_settings, save_session_settings,
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, send_whatsapp_message,
    normalize_language_input
)

app = FastAPI()

# Twilio WhatsApp Endpoints

@app.api_route("/twilio/whatsapp", methods=["POST"])
async def twilio_whatsapp(request: Request):
    form_data = await request.form()
    from_number = form_data.get("From")
    body = form_data.get("Body", "").strip()
    media_url = form_data.get("MediaUrl0")
    content_type = form_data.get("MediaContentType0", "")

    # Get or initialize session settings (using phone number as session_id)
    session_id = from_number
    settings = get_session_settings(session_id)
    current_lang = settings.get("lang")  # None if not set

    response = MessagingResponse()

    # 1. Check for language switch command (accepts full names or codes)
    if body and not media_url:
        lang_code = normalize_language_input(body)
        
        if lang_code:
            settings["lang"] = lang_code
            save_session_settings(session_id, settings)
            
            lang_confirmations = {
                'en': 'Language set to: English 🇬🇧',
                'ml': 'Language set to: Malayalam 🇮🇳',
                'hi': 'Language set to: Hindi 🇮🇳',
                'ta': 'Language set to: Tamil 🇮🇳',
                'ar': 'Language set to: Arabic 🇸🇦',
                'fr': 'Language set to: French 🇫🇷',
                'es': 'Language set to: Spanish 🇪🇸',
            }
            
            confirmation = lang_confirmations.get(lang_code, f'Language set to: {lang_code.upper()}')
            response.message(f"{confirmation}\n\nYou can now send a voice note and I'll respond!")
            return Response(content=str(response), media_type="application/xml")

    # 2. If no language is set, force selection
    if not current_lang:
        welcome_msg = (
            "🎙️ Welcome to the Voice AI Bot!\n\n"
            "Please select your preferred language:\n\n"
            "🇬🇧 English (type: en or english)\n"
            "🇮🇳 Malayalam (type: ml or malayalam)\n"
            "🇮🇳 Hindi (type: hi or hindi)\n"
            "🇮🇳 Tamil (type: ta or tamil)\n"
            "🇸🇦 Arabic (type: ar or arabic)\n"
            "🇫🇷 French (type: fr or french)\n"
            "🇪🇸 Spanish (type: es or spanish)\n\n"
            "Just type the language name or code to begin."
        )
        response.message(welcome_msg)
        return Response(content=str(response), media_type="application/xml")

    # 3. Check for voice note (Media message)
    if media_url and "audio" in content_type:
        print(f"[*] WhatsApp: Downloading media from {media_url}")
        
        # Download the audio with Auth
        temp_input = f"{TEMP_DIR}/{uuid.uuid4()}.ogg"  # WhatsApp usually sends .ogg
        
        try:
            media_response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            
            if media_response.status_code != 200:
                print(f"[!] WhatsApp: Failed to download media: {media_response.status_code}")
                response.message("❌ Error downloading voice note. Please try again.")
                return Response(content=str(response), media_type="application/xml")

            with open(temp_input, "wb") as f:
                f.write(media_response.content)

            # Preprocess
            print("[*] WhatsApp: Preprocessing audio...")
            clean_input = f"{TEMP_DIR}/{uuid.uuid4()}_clean.wav"
            cleaning_success = clean_audio_gentle(temp_input, clean_input)
            audio_file = clean_input if cleaning_success else temp_input

            # Transcribe
            print(f"[*] WhatsApp: Transcribing in {current_lang}...")
            user_text = transcribe_audio(audio_file, current_lang)
            print(f"[*] WhatsApp: User said: {user_text}")

            # Handle transcription failures
            if user_text.startswith("["):
                response.message("⚠️ Sorry, I couldn't understand the audio. Please try again with clearer audio.")
                return Response(content=str(response), media_type="application/xml")

            # AI Response
            print("[*] WhatsApp: Generating AI response...")
            bot_reply = generate_ai_response(user_text, session_id, current_lang)
            
            if not bot_reply:
                bot_reply = "I'm sorry, I couldn't generate a response."
            
            print(f"[*] WhatsApp: Bot reply: {bot_reply}")
            
            # Save History
            save_chat_message(session_id, user_text, bot_reply, current_lang)

            # TTS for WhatsApp Voice Note
            output_audio_filename = f"{uuid.uuid4()}.mp3"
            output_audio_path = f"{TEMP_DIR}/{output_audio_filename}"
            
            tts_success = text_to_speech(bot_reply, current_lang, output_audio_path)
            
            # Get public URL for the audio
            base_url = str(request.base_url).rstrip("/")
            audio_url = f"{base_url}/audio/{output_audio_filename}"

            # Send via REST API with Media (as a Voice Note)
            if tts_success:
                api_success = send_whatsapp_message(from_number, bot_reply, media_url=audio_url)
                print(f"[*] WhatsApp: API send success: {api_success}")
            else:
                print("[!] WhatsApp: TTS failed, sending text only")
                send_whatsapp_message(from_number, bot_reply)
            
            # Return TwiML (fallback)
            response.message(bot_reply)
            
            return Response(content=str(response), media_type="application/xml")
            
        except Exception as e:
            print(f"[!] WhatsApp: Processing error: {e}")
            response.message("❌ An error occurred processing your voice note. Please try again.")
            return Response(content=str(response), media_type="application/xml")
        
        finally:
            # Cleanup temporary files
            try:
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                if 'clean_input' in locals() and os.path.exists(clean_input):
                    os.remove(clean_input)
            except Exception as e:
                print(f"[!] Cleanup error: {e}")

    # 4. Default response for text messages (not language selection)
    if body and not media_url:
        response.message(
            f"💬 I currently only process voice notes.\n\n"
            f"📱 Your current language: {current_lang.upper()}\n\n"
            f"To change language, type:\n"
            f"• 'en' for English\n"
            f"• 'ml' for Malayalam\n"
            f"• 'hi' for Hindi\n"
            f"• 'ta' for Tamil\n"
            f"• etc."
        )
    else:
        response.message("🎤 Send me a voice note and I'll respond in your language!")

    return Response(content=str(response), media_type="application/xml")

@app.get("/audio/{filename}")
def get_audio(filename: str):
    """Serve audio files for WhatsApp"""
    file_path = f"{TEMP_DIR}/{filename}"
    
    if not os.path.exists(file_path):
        return Response(content="File not found", status_code=404)
    
    return FileResponse(file_path, media_type="audio/mpeg")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "whatsapp-voice-bot"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)