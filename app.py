import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "gemini-3.1-flash-lite"

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured in the .env file.")

client = genai.Client(api_key=api_key)

config_path = BASE_DIR / "chatbot_config"
SYSTEM_PROMPT = config_path.read_text(encoding="utf-8").strip()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.35,
                max_output_tokens=700,
            ),
        )

        answer = (response.text or "").strip()

        if not answer:
            answer = "I couldn't generate a response right now. Please try again."

        return jsonify({"answer": answer})

    except Exception:
        app.logger.exception("Gemini API request failed")
        return jsonify({
            "error": "The assistant is temporarily unavailable. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
