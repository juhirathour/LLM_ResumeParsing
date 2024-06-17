import os
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PyPDF2 import PdfFileReader

BASE_UPLOAD_PATH = "__DATA__"
if not os.path.exists(BASE_UPLOAD_PATH):
    os.makedirs(BASE_UPLOAD_PATH)

def load_model_and_tokenizer(model_name, huggingface_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=huggingface_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=huggingface_token)
    return tokenizer, model

def read_file_from_path(path):
    reader = PdfFileReader(path)
    data = ""

    for page_no in range(len(reader.pages)):
        page = reader.getPage(page_no)
        data += page.extract_text()

    return data

def ats_extractor(data, tokenizer, model):
    inputs = tokenizer(data, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def save_uploaded_file(upload, upload_path):
    unique_dir = os.path.join(upload_path, str(uuid.uuid4()))
    os.makedirs(unique_dir, exist_ok=True)
    doc_path = os.path.join(unique_dir, "file.pdf")
    upload.save(doc_path)
    return doc_path
