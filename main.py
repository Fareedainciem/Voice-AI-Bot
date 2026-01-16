import os
import uuid
import warnings
import json
import sys
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
import whisperx
from openai import OpenAI
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import torch

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

# Configure OpenAI API
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
    print("OpenAI GPT-4O Mini configured successfully")
else:
    print("ERROR: OPENAI_API_KEY not found!")
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Load Whisper X model - Use large for best accuracy
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"
whisper_model = whisperx.load_model("large", device=device, compute_type=compute_type, vad_method="silero")
print(f"WhisperX 'large' loaded on {device.upper()} (compute_type: {compute_type})")

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Chat history storage
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

def load_chat_history(session_id):
    """Load chat history for a session"""
    history_file = f"{CHAT_HISTORY_DIR}/{session_id}.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_chat_message(session_id, user_text, bot_reply, language):
    """Save a message to chat history"""
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
    """Enhanced audio preprocessing for accuracy"""
    try:
        # Load audio at 16kHz
        y, sr = librosa.load(input_path, sr=16000)

        # Aggressive trim of silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Multi-pass noise reduction for clarity
        if len(y) > int(0.1 * sr):
            noise_sample = y[:int(0.1 * sr)]
            # Two passes for better noise removal
            y = nr.reduce_noise(y=y, y_noise=noise_sample, sr=sr, prop_decrease=0.8)
            y = nr.reduce_noise(y=y, y_noise=noise_sample, sr=sr, prop_decrease=0.6)
        
        # Normalize and boost for clarity
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val * 0.9

        # Save cleaned audio
        sf.write(output_path, y, sr)
        return True
    except Exception as e:
        print(f"Audio cleaning error: {e}")
        return False

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

        # ALWAYS transcribe in selected language
        # Then respond in user's selected language
        response_language_requested = language  # Store user's language preference
        
        # Direct language mode (e.g., ml, hi, ta, etc.)
        # Map language codes to Whisper codes
        lang_map = {'en': 'en', 'ml': 'ml', 'hi': 'hi', 'ta': 'ta', 'ar': 'ar', 'fr': 'fr', 'es': 'es'}
        transcribe_lang = lang_map.get(language, 'en')
        print(f"Direct mode: Transcribing in selected language: {language} (Whisper code: {transcribe_lang})")
        print(f"Will output text in {language} and respond in {language}")

        # Language-specific prompts to anchor Whisper to the correct script
        initial_prompts = {
            'en': 'English.',
            'ml': 'മലയാളം. ഈ നാട്ടിലെ ആളുകൾ മലയാളം സംസാരിക്കുന്നു.',
            'hi': 'हिंदी. यह भारत में बोली जाती है।',
            'ta': 'தமிழ். இது தமிழ் மொழியாகும்.',
            'ar': 'العربية.',
            'fr': 'Français.',
            'es': 'Español.'
        }

        current_prompt = initial_prompts.get(transcribe_lang, None)

        # Transcribe with language-optimized settings
        transcribe_options = {
            'fp16': False,
            'verbose': False,
            'best_of': 2,
            'beam_size': 3,
            'patience': 1.0,
            'length_penalty': 1.0,
            'compression_ratio_threshold': 2.4,
            'logprob_threshold': -1.0,
            'no_speech_threshold': 0.3,
            'condition_on_previous_text': False,
            'temperature': 0.2
        }

        task = 'transcribe'
        print(f"Task: transcribe (will transcribe in {transcribe_lang} language)")

        # WhisperX transcription with language enforcement
        try:
            # Check if audio file exists and has size
            if not os.path.exists(audio_to_transcribe) or os.path.getsize(audio_to_transcribe) == 0:
                print(f"ERROR: Audio file {audio_to_transcribe} is missing or empty")
                user_text = "[Audio error]"
            else:
                # Use smaller batch size on CPU to prevent crashes
                effective_batch_size = 4 if device == "cpu" else 16
                
                result = whisper_model.transcribe(
                    audio_to_transcribe,
                    language=transcribe_lang,
                    task=task,
                    batch_size=effective_batch_size,
                    print_progress=False
                )
            
            print(f"DEBUG: Whisper transcribed with language={transcribe_lang}, task={task}")
            
            # WhisperX returns different format - extract text from segments
            if isinstance(result, dict):

                if "text" in result:
                    user_text = result["text"].strip()
                elif "segments" in result:
                    # Extract text from segments
                    user_text = " ".join([seg.get("text", "") for seg in result["segments"]]).strip()
                else:
                    print(f"DEBUG: Result keys = {result.keys()}")
                    user_text = str(result)
            else:
                user_text = str(result)
                
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            user_text = "[Transcription failed]"
        
        # Determine detected language based on transcribe settings
        detected_language = transcribe_lang or "en"
        language_probability = 1.0
        
        print("="*50)
        print(f"TRANSCRIPTION RESULT:")
        print(f"  Selected: {language}")
        print(f"  Detected: {detected_language}")
        print(f"  Confidence: {language_probability:.2%}")
        # Safely print text for console
        try:
            print(f"  Text: {user_text}")
        except:
            print(f"  Text: [Contains non-ASCII characters]")
        print("="*50)

        # Determine response language based on user's selection
        language_names = {
            'en': 'English',
            'ml': 'Malayalam',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'ar': 'Arabic',
            'fr': 'French',
            'es': 'Spanish'
        }
        
        # Use user's requested language for response, default to English
        response_language = language_names.get(response_language_requested, 'English')
        tts_lang = response_language_requested
        print(f"Using requested language: {response_language_requested} -> {response_language}")

        # Generate response with GPT-4O Mini
        try:
            # Load chat history for context
            chat_history = load_chat_history(session_id) if session_id else []
            
            # Build conversation context from last 5 messages
            system_message = f"""You are an accurate, helpful multilingual assistant. Your response MUST be in {response_language} language only.
            
Rules:
1. Respond ONLY in {response_language}
2. Do not translate or code-switch to other languages
3. Be concise and accurate
4. If the question is unclear, ask for clarification in {response_language}
5. Keep responses under 200 characters when possible"""

            context_messages = [{"role": "system", "content": system_message}]
            
            # Add recent history (last 5 exchanges)
            for msg in chat_history[-5:]:
                context_messages.append({"role": "user", "content": msg["user"]})
                context_messages.append({"role": "assistant", "content": msg["bot"]})
            
            # Add current message
            context_messages.append({"role": "user", "content": user_text})
            
            print(f"DEBUG: Sending {len(context_messages)} messages to OpenAI")
            print(f"DEBUG: Response language: {response_language}")
            print(f"DEBUG: System prompt: {system_message[:100]}...")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=context_messages,
                max_tokens=200,
                temperature=0.3,
            )
            bot_reply = response.choices[0].message.content.strip()
            print(f"DEBUG: Got response from OpenAI in {response_language}: {bot_reply[:100]}...")
        except Exception as e:
            print(f"OpenAI error: {e}")
            import traceback
            traceback.print_exc()
            bot_reply = f"I apologize, but I couldn't generate a response. Error: {str(e)}"

        # Text to Speech
        output_audio_path = f"{TEMP_DIR}/{uuid.uuid4()}.mp3"
        
        # Save chat message to history
        if session_id:
            save_chat_message(session_id, user_text, bot_reply, response_language_requested or "en")
        
        # Map language codes for gTTS
        gtts_lang_map = {
            'en': 'en',
            'ml': 'ml',
            'hi': 'hi',
            'ta': 'ta',
            'ar': 'ar',
            'fr': 'fr',
            'es': 'es'
        }
        
        gtts_lang = gtts_lang_map.get(tts_lang, 'en')
        
        try:
            # Use appropriate TLD for better pronunciation
            tld_map = {
                'ml': 'co.in',
                'hi': 'co.in',
                'ta': 'co.in',
                'en': 'com',
                'ar': 'com.sa',
                'fr': 'fr',
                'es': 'es'
            }
            tld = tld_map.get(tts_lang, 'com')
            
            tts = gTTS(text=bot_reply, lang=gtts_lang, tld=tld, slow=False)
            tts.save(output_audio_path)
        except Exception as e:
            print(f"gTTS error: {e}")
            # Fallback to English
            tts = gTTS(text="Sorry, I couldn't generate audio response.", lang='en', slow=False)
            tts.save(output_audio_path)

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