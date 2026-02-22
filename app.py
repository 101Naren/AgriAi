from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import tempfile

app = Flask(__name__)
CORS(app)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

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


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    if not data:
        return jsonify({"response": "Invalid JSON body.", "success": False}), 400

    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "No message provided.", "success": False}), 400

    full_prompt = f"{AGRICULTURE_CONTEXT}\n\nUser Question: {user_message}\n\nAnswer:"

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False},
            timeout=60,
        )

        if resp.status_code == 200:
            result = resp.json()
            answer = result.get("response", "").strip()
            if not answer:
                return jsonify({"response": "Empty response from model.", "success": False}), 500
            return jsonify({"response": answer, "success": True})
        else:
            return jsonify({"response": f"Ollama error {resp.status_code}: {resp.text}", "success": False}), 500

    except requests.exceptions.ConnectionError:
        return jsonify({
            "response": "Cannot connect to Ollama. Run: ollama serve",
            "success": False
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({"response": "Request timed out. Model may be loading.", "success": False}), 504
    except Exception as e:
        return jsonify({"response": f"Unexpected error: {str(e)}", "success": False}), 500


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided.", "success": False}), 400

    audio_file = request.files["audio"]
    content_type = audio_file.content_type or "audio/webm"

    # Determine file extension from MIME type
    ext_map = {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/mp4": ".mp4",
        "audio/mpeg": ".mp3",
    }
    ext = ext_map.get(content_type, ".webm")

    # --- Option 1: OpenAI Whisper API (if key is set) ---
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            audio_bytes = audio_file.read()
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=(f"audio{ext}", audio_bytes, content_type),
            )
            return jsonify({"transcript": transcript.text, "success": True, "engine": "openai-whisper"})
        except Exception as e:
            return jsonify({"error": str(e), "success": False}), 500

    # --- Option 2: Local openai-whisper package ---
    try:
        import whisper

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path)
            transcript_text = result["text"].strip()
        finally:
            os.unlink(tmp_path)

        return jsonify({"transcript": transcript_text, "success": True, "engine": "local-whisper"})

    except ImportError:
        return jsonify({
            "error": "Transcription unavailable. Either set OPENAI_API_KEY or run: pip install openai-whisper",
            "success": False,
            "fallback": True
        }), 501
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/health", methods=["GET"])
def health():
    ollama_ok = False
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    return jsonify({
        "status": "ok",
        "ollama": "running" if ollama_ok else "not running",
        "model": OLLAMA_MODEL
    })


if __name__ == "__main__":
    print("=" * 50)
    print("AgriAI Backend Starting...")
    print(f"  Model  : {OLLAMA_MODEL}")
    print(f"  API    : http://localhost:5000")
    print(f"  Ollama : {OLLAMA_URL}")
    print("=" * 50)
    print("Tip: set OPENAI_API_KEY env var to enable cloud Whisper transcription")
    print("     OR install openai-whisper for local transcription: pip install openai-whisper")
    print("=" * 50)
    app.run(debug=True, port=5000)
