import os
import sys
import yaml
import uuid
from flask import Flask, request, render_template, jsonify
import json
from huggingface_hub import login
import resumeparser
from transformers import AutoConfig, AutoModelForSequenceClassification

sys.path.insert(0, os.path.abspath(os.getcwd()))

app = Flask(__name__)

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
HUGGINGFACE_TOKEN = config.get("HUGGINGFACE_TOKEN")

# Authenticate with Hugging Face
login(HUGGINGFACE_TOKEN)
print("Login successful")

# Load the model and tokenizer from Hugging Face
tokenizer, model = resumeparser.load_model_and_tokenizer(MODEL_NAME, HUGGINGFACE_TOKEN)
print("Model and tokenizer loaded")

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/process", methods=["POST"])
def process_resume():
    if 'pdf_doc' not in request.files:
        return "No file part", 400
    doc = request.files['pdf_doc']
    if doc.filename == '':
        return "No selected file", 400

    # Save uploaded file
    doc_path = resumeparser.save_uploaded_file(doc, resumeparser.BASE_UPLOAD_PATH)
    text_data = resumeparser.read_file_from_path(doc_path)
    ats_data = resumeparser.ats_extractor(text_data, tokenizer, model)

    return render_template('index.html', data=json.loads(ats_data))

if __name__ == "__main__":
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Disable symlinks warning
    app.run(port=8000, debug=True)
