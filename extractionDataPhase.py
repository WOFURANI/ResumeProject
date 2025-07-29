import os
import re
import cv2
import pytesseract
import fitz
import spacy
from pathlib import Path
from mimetypes import guess_type


nlp = spacy.load("en_core_web_sm")

SKILLS = [
    "Python", "Java", "C++", "SQL", "JavaScript", "HTML", "CSS",
    "React", "Node.js", "Machine Learning", "Data Analysis"
]

def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r'\+?\d[\d\s\-()]{8,}\d', text)
    return match.group(0) if match else None

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_skills(text, skill_keywords):
    found = []
    for skill in skill_keywords:
        if re.search(rf'\b{re.escape(skill)}\b', text, re.IGNORECASE):
            found.append(skill)
    return list(set(found))

def parse_resume(file_path):
    mime, _ = guess_type(file_path)
    text = ""
    if mime:
        if "pdf" in mime:
            text = extract_text_from_pdf(file_path)
        elif "image" in mime:
            text = extract_text_from_image(file_path)
    else:
        ext = Path(file_path).suffix.lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            text = extract_text_from_image(file_path)
        elif ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        else:
            raise ValueError("Unsupported file type")

    result = {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Phone": extract_phone(text),
        "Skills": extract_skills(text, SKILLS)
    }

    return result

if __name__ == "__main__":
    file = "resume.pdf"
    data = parse_resume(file)
    print("Extracted Resume Info:\n", data)
