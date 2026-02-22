from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json
import sys

app = Flask(__name__)
CORS(app)

# Agriculture knowledge base
AGRICULTURE_CONTEXT = """You are an expert agriculture assistant helping farmers. Provide practical, actionable advice.

Key Topics:
- Rice: Needs standing water, NPK 4:2:1, pH 5.5-7.0
- Wheat: Well-drained soil, drip irrigation, NPK 4:2:1
- Tomato: Protect from blight with copper fungicides, needs regular watering
- Cotton: Black soil, moderate irrigation, IPM for bollworms

Diseases:
- Blight: Use copper fungicides, crop rotation
- Rust: Sulfur fungicides, resistant varieties
- Wilt: Soil solarization, proper drainage

Fertilizers:
- Nitrogen (Urea): For leaf growth
- Phosphorus (DAP): For roots
- Potassium (MOP): For disease resistance

Answer in the same language the farmer asks. Be concise and practical."""

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    # Create prompt for Llama
    full_prompt = f"{AGRICULTURE_CONTEXT}\n\nUser Question: {user_message}\n\nAnswer:"
    
    try:
        # Try different Ollama command formats for Windows
        commands = [
            ['ollama', 'run', 'llama3.2:3b', full_prompt],
            ['ollama.exe', 'run', 'llama3.2:3b', full_prompt],
            ['cmd', '/c', 'ollama', 'run', 'llama3.2:3b', full_prompt]
        ]
        
        response_text = None
        last_error = None
        
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    shell=False
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    response_text = result.stdout.strip()
                    break
                else:
                    last_error = result.stderr
                    
            except Exception as e:
                last_error = str(e)
                continue
        
        if response_text:
            return jsonify({
                'response': response_text,
                'success': True
            })
        else:
            return jsonify({
                'response': f"Could not connect to Ollama. Make sure Ollama is running. Error: {last_error}",
                'success': False
            }), 500
    
    except Exception as e:
        return jsonify({
            'response': f"Error: {str(e)}",
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("=" * 50)
    print("Agriculture AI Backend Starting...")
    print("API running at http://localhost:5000")
    print("=" * 50)
    print("\nMake sure Ollama is running!")
    print("Test with: ollama list")
    print("=" * 50)
    app.run(debug=True, port=5000)