import os
import uuid
import json
import torch
import whisperx
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from datetime import datetime
from openai import OpenAI
from gtts import gTTS
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

# Configure OpenAI
openai_client = None
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    openai_client = OpenAI(api_key=api_key)
else:
    print("WARNING: OPENAI_API_KEY not found!")

# Configure Twilio Credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    print(f"Twilio credentials loaded successfully (Account SID: {TWILIO_ACCOUNT_SID[:5]}...)")
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
else:
    print("WARNING: Twilio credentials not found in .env!")

# Language configuration mapping
LANGUAGE_MAP = {
    'english': 'en', 'en': 'en',
    'malayalam': 'ml', 'ml': 'ml',
    'hindi': 'hi', 'hi': 'hi',
    'tamil': 'ta', 'ta': 'ta',
    'arabic': 'ar', 'ar': 'ar',
    'french': 'fr', 'fr': 'fr',
    'spanish': 'es', 'es': 'es',
}

def normalize_language_input(user_input):
    """Convert user language input to language code"""
    normalized = user_input.lower().strip()
    return LANGUAGE_MAP.get(normalized)

def send_whatsapp_message(to_number, message_body, media_url=None):
    """Fallback method using Twilio REST API"""
    if not twilio_client:
        print("[!] Twilio client not initialized")
        return False
    print(f"[*] Twilio API: Attempting to send from {TWILIO_WHATSAPP_NUMBER} to {to_number} (Media: {media_url is not None})")
    try:
        msg_params = {
            "from_": TWILIO_WHATSAPP_NUMBER,
            "body": message_body,
            "to": to_number
        }
        if media_url:
            msg_params["media_url"] = [media_url]
            
        message = twilio_client.messages.create(**msg_params)
        print(f"[*] Sent WhatsApp message via API: {message.sid}")
        return True
    except Exception as e:
        print(f"[!] Twilio API Error: {e}")
        return False

# Whisper Model Configuration (Lazy Loaded)
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"
        _whisper_model = whisperx.load_model("medium", device=device, compute_type=compute_type, vad_method="silero")
        print(f"WhisperX 'medium' loaded on {device.upper()} (compute_type: {compute_type})")
    return _whisper_model

def detect_script(text):
    """Detect which script(s) are present in the text"""
    scripts = []
    
    # Unicode ranges for common Indic scripts
    script_ranges = {
        'Devanagari': ('\u0900', '\u097F'),
        'Malayalam': ('\u0D00', '\u0D7F'),
        'Tamil': ('\u0B80', '\u0BFF'),
        'Telugu': ('\u0C00', '\u0C7F'),
        'Kannada': ('\u0C80', '\u0CFF'),
        'Bengali': ('\u0980', '\u09FF'),
        'Gujarati': ('\u0A80', '\u0AFF'),
        'Arabic': ('\u0600', '\u06FF'),
        'Latin': ('\u0020', '\u007F'),
    }
    
    for script_name, (start, end) in script_ranges.items():
        if any(start <= c <= end for c in text):
            scripts.append(script_name)
    
    return scripts

def fix_script_mismatch(text, target_lang):
    """
    Fix script mismatches by transliterating from Devanagari to target script.
    Requires: pip install indic-transliteration
    """
    if not text or target_lang == 'hi':
        return text
    
    # Check if transliteration library is available
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        print("[!] Warning: indic-transliteration not installed. Install with: pip install indic-transliteration")
        return text
    
    script_map = {
        'ml': sanscript.MALAYALAM,
        'ta': sanscript.TAMIL,
        'te': sanscript.TELUGU,
        'kn': sanscript.KANNADA,
        'bn': sanscript.BENGALI,
        'gu': sanscript.GUJARATI,
    }
    
    # Only transliterate if target is an Indic language and text contains Devanagari
    if target_lang in script_map:
        detected_scripts = detect_script(text)
        
        if 'Devanagari' in detected_scripts and detected_scripts != [target_lang.upper()]:
            print(f"[*] Script mismatch detected: {detected_scripts}. Transliterating to {target_lang.upper()}")
            try:
                transliterated = transliterate(text, sanscript.DEVANAGARI, script_map[target_lang])
                print(f"[*] Transliterated: {text} -> {transliterated}")
                return transliterated
            except Exception as e:
                print(f"[!] Transliteration failed: {e}")
                return text
    
    return text

def is_correct_script(text, lang_code):
    """Validate if text is in the expected script for the language"""
    expected_scripts = {
        'ml': 'Malayalam',
        'ta': 'Tamil',
        'hi': 'Devanagari',
        'te': 'Telugu',
        'kn': 'Kannada',
        'bn': 'Bengali',
        'gu': 'Gujarati',
        'ar': 'Arabic',
        'en': 'Latin',
        'fr': 'Latin',
        'es': 'Latin',
    }
    
    if lang_code not in expected_scripts:
        return True  # Unknown language, assume valid
    
    expected = expected_scripts[lang_code]
    detected = detect_script(text)
    
    return expected in detected

def transcribe_audio(audio_path, lang_code):
    """Transcribe audio using WhisperX with script validation and correction"""
    try:
        model = get_whisper_model()
        
        result = model.transcribe(
            audio_path,
            language=lang_code,
            task='transcribe',
            batch_size=4 if not torch.cuda.is_available() else 16
        )
        
        user_text = ""
        if isinstance(result, dict):
            if "text" in result and result["text"].strip():
                user_text = result["text"].strip()
            elif "segments" in result:
                user_text = " ".join([seg.get("text", "") for seg in result["segments"]]).strip()
        
        if not user_text:
            return "[Silence or unclear speech]"
        
        # Debug: Log detected scripts
        detected_scripts = detect_script(user_text)
        print(f"[*] Transcription: '{user_text}'")
        print(f"[*] Detected scripts: {detected_scripts}")
        print(f"[*] Expected language: {lang_code}")
        
        # Validate and fix script if needed
        if not is_correct_script(user_text, lang_code):
            print(f"[!] Script mismatch! Attempting to fix...")
            user_text = fix_script_mismatch(user_text, lang_code)
            
            # Re-validate
            if is_correct_script(user_text, lang_code):
                print(f"[✓] Script corrected successfully")
            else:
                print(f"[!] Script correction failed, proceeding with original text")
        
        return user_text
        
    except Exception as e:
        print(f"[!] Transcribe Error: {e}")
        return "[Transcription failed]"

TEMP_DIR = "temp_audio"
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

def get_session_settings(session_id):
    settings_file = f"{CHAT_HISTORY_DIR}/{session_id}_settings.json"
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_session_settings(session_id, settings):
    settings_file = f"{CHAT_HISTORY_DIR}/{session_id}_settings.json"
    with open(settings_file, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

def load_chat_history(session_id):
    history_file = f"{CHAT_HISTORY_DIR}/{session_id}.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_chat_message(session_id, user_text, bot_reply, language):
    history_file = f"{CHAT_HISTORY_DIR}/{session_id}.json"
    history = load_chat_history(session_id)
    history.append({
        "timestamp": datetime.now().isoformat(),
        "user": user_text,
        "bot": bot_reply,
        "language": language
    })
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return history

def clean_audio_gentle(input_path, output_path):
    print(f"[*] Audio: Loading {input_path}")
    try:
        y, sr = librosa.load(input_path, sr=16000)
        print(f"[*] Audio: Loaded {len(y)} samples at {sr}Hz")
        if len(y) == 0:
            print("[!] Audio: Loaded file is empty!")
            return False
            
        y, _ = librosa.effects.trim(y, top_db=20)
        if len(y) > int(0.1 * sr):
            noise_sample = y[:int(0.1 * sr)]
            y = nr.reduce_noise(y=y, y_noise=noise_sample, sr=sr, prop_decrease=0.8)
            y = nr.reduce_noise(y=y, y_noise=noise_sample, sr=sr, prop_decrease=0.6)
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val * 0.9
        sf.write(output_path, y, sr)
        print(f"[*] Audio: Saved to {output_path}")
        return True
    except Exception as e:
        print(f"[!] Audio cleaning error: {e}")
        return False

def generate_ai_response(user_text, session_id, lang_code):
    chat_history = load_chat_history(session_id) if session_id else []
    
    # Language names for better context
    lang_names = {
        'en': 'English',
        'ml': 'Malayalam',
        'hi': 'Hindi',
        'ta': 'Tamil',
        'ar': 'Arabic',
        'fr': 'French',
        'es': 'Spanish',
    }
    
    lang_name = lang_names.get(lang_code, lang_code.upper())
    
    # Updated system message for factual rigor and native script
    system_message = f"""You are a highly accurate professional research assistant and polyglot. 
The user is communicating in {lang_name} (language code: '{lang_code}').

STRICT OPERATING PROCEDURES:
1. FACTUAL RESEARCH: You must be 100% factually accurate. Do not hallucinate, guess, or provide incorrect geographic/scientific data. If you are not absolutely sure about a fact (like the longest river or its origin), say you don't know rather than guessing.
2. NATIVE SCRIPT: Your entire response MUST be in the native script of '{lang_code}'. Never use Latin script for native responses.
3. TRANSCRIPTION RECOVERY: The user's input may be provided as a phonetic or translated transcription (e.g., English text for a Malayalam question). You must decode the intended meaning and provide the answer in the native script of '{lang_code}'.
4. PROFESSIONALISM: Keep your response concise (under 300 characters), professional, and informative."""

    context_messages = [{"role": "system", "content": system_message}]
    
    # Include last 5 messages for context
    for msg in chat_history[-5:]:
        context_messages.append({"role": "user", "content": msg["user"]})
        context_messages.append({"role": "assistant", "content": msg["bot"]})
    
    context_messages.append({"role": "user", "content": user_text})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context_messages,
            max_tokens=400,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[!] OpenAI API Error: {e}")
        return "I apologize, I encountered an error generating a response."

def text_to_speech(text, lang_code, output_path):
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"[!] TTS Error: {e}")
        # Fallback to a simple error message
        return None