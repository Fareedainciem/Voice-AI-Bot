from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import os
import uuid

app = Flask(__name__)
BACKEND_URL = "http://127.0.0.1:8000"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        # Get audio file and language
        audio_file = request.files.get('audio')
        language = request.form.get('language')
        
        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400
        
        # Save temporarily
        filename = f"{uuid.uuid4()}.wav"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(filepath)
        
        # Send to backend
        files = {'audio': open(filepath, 'rb')}
        data = {}
        if language:
            data['language'] = language
        
        response = requests.post(
            f"{BACKEND_URL}/voice-chat",
            files=files,
            data=data,
            timeout=600
        )
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(response.json())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)