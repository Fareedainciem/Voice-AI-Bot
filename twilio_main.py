import os
import uuid
import requests
import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import Response, FileResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from voice_logic import (
    get_whisper_model, transcribe_audio, generate_ai_response, text_to_speech, 
    clean_audio_gentle, save_chat_message, TEMP_DIR,
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
)

app = FastAPI()

# Twilio Voice Endpoints

@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice(request: Request):
    """Initial greeting and language selection or direct recording"""
    vr = VoiceResponse()
    
    # You can customize this to ask for language first, 
    # but for now, we'll assume English or use a query param
    language = request.query_params.get("lang", "en")
    
    vr.say(f"Please speak after the beep. I will respond in {language}.", language="en-US")
    
    # Record the user's voice
    vr.record(
        action=f"/twilio/process?lang={language}",
        max_length=15,
        play_beep=True,
        timeout=5
    )
    
    return Response(content=str(vr), media_type="application/xml")

@app.api_route("/twilio/process", methods=["GET", "POST"])
async def twilio_process(request: Request, lang: str = "en"):
    """Handle the recording from Twilio"""
    form_data = await request.form()
    recording_url = form_data.get("RecordingUrl")
    call_sid = form_data.get("CallSid")
    
    vr = VoiceResponse()
    
    if not recording_url:
        vr.say("I didn't hear anything. Goodbye.")
        return Response(content=str(vr), media_type="application/xml")

    # 1. Download the recording with Auth
    print(f"[*] Downloading recording from {recording_url}")
    temp_input = f"{TEMP_DIR}/{uuid.uuid4()}.wav"
    response = requests.get(recording_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    if response.status_code != 200:
        print(f"[!] Failed to download recording: {response.status_code}")
        vr.say("Error downloading recording.")
        return Response(content=str(vr), media_type="application/xml")
    
    with open(temp_input, "wb") as f:
        f.write(response.content)

    # 2. Preprocess
    print("[*] Preprocessing audio...")
    clean_input = f"{TEMP_DIR}/{uuid.uuid4()}_clean.wav"
    cleaning_success = clean_audio_gentle(temp_input, clean_input)
    audio_file = clean_input if cleaning_success else temp_input

    # 3. Transcribe
    print(f"[*] Transcribing in {lang}...")
    user_text = transcribe_audio(audio_file, lang)
    print(f"[*] User said: {user_text}")

    # 4. AI Response
    print("[*] Generating AI response...")
    bot_reply = generate_ai_response(user_text, call_sid, lang)
    print(f"[*] Bot reply: {bot_reply}")
    
    # 5. Save History
    save_chat_message(call_sid, user_text, bot_reply, lang)

    # 6. TTS
    output_audio_filename = f"{uuid.uuid4()}.mp3"
    output_audio_path = f"{TEMP_DIR}/{output_audio_filename}"
    text_to_speech(bot_reply, lang, output_audio_path)

    # 7. Play back to Twilio
    # Note: Twilio needs a PUBLIC URL. The user will need to configure their Base URL.
    # We'll use a placeholder or relative path if Twilio is configured with the base.
    base_url = str(request.base_url).rstrip("/")
    audio_url = f"{base_url}/audio/{output_audio_filename}"
    
    vr.play(audio_url)
    
    # Loop back to record more
    vr.redirect(f"/twilio/voice?lang={lang}")
    
    return Response(content=str(vr), media_type="application/xml")

@app.get("/audio/{filename}")
def get_audio(filename: str):
    return FileResponse(f"{TEMP_DIR}/{filename}", media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
