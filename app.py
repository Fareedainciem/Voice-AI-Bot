import streamlit as st
import requests
from streamlit_mic_recorder import mic_recorder

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Voice AI Bot", page_icon="🎤")
st.title("🎤 Multilingual Voice Bot")
st.write("Record your voice or upload an audio file to chat with the bot.")

# Language selection
st.subheader("🌍 Select Language")
language_options = {
    "English": "en",
    "Malayalam (മലയാളം)": "ml",
    "Hindi (हिंदी)": "hi",
    "Arabic": "ar",
    "French": "fr",
    "Spanish": "es",
    "Tamil (தமிழ்)": "ta"
}

selected_language_name = st.selectbox(
    "Choose the language you'll speak in:",
    options=list(language_options.keys()),
    index=0
)
selected_language_code = language_options[selected_language_name]

st.divider()

# Create tabs for different input methods
tab1, tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload Audio"])

audio_bytes = None

with tab1:
    st.write("Click the microphone to start recording, speak, then stop.")
    audio = mic_recorder(key="recorder")
    
    if audio is not None:
        audio_bytes = audio["bytes"]
        st.audio(audio_bytes, format="audio/wav")

with tab2:
    st.write("Upload an audio file (WAV, MP3, M4A, etc.)")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])
    
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes)

# Process audio if available
if audio_bytes is not None:
    # Only process when user actually has audio (recording or upload), not on language change
    st.info(f"Processing your voice... (Language: {selected_language_name})")

    files = {
        "audio": ("recording.wav", audio_bytes, "audio/wav")
    }
    
    # Add language parameter
    data_params = {}
    if selected_language_code:
        data_params["language"] = selected_language_code

    try:
        response = requests.post(
            f"{BACKEND_URL}/voice-chat",
            files=files,
            data=data_params,
            timeout=600
        )
        data = response.json()

        st.subheader("🧾 You said")
        st.write(data["user_text"])

        st.subheader("🤖 Bot replied (text)")
        st.write(data["bot_text"])

        st.subheader("🔊 Bot replied (voice)")
        st.audio(f"{BACKEND_URL}{data['audio_url']}")
        
        # Clear audio after processing
        st.session_state.audio_processed = True

    except Exception as e:
        st.error(f"Backend error: {e}")
