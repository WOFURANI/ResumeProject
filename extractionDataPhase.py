import os
import re
import cv2
import fitz
import spacy
import pytesseract
import nltk
import sqlite3
from PIL import Image
from spacy.training import Example
from spacy.scorer import Scorer
from collections import defaultdict

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

nlp = spacy.load("custom_ner_model")

# Define keyword lists
SKILL_KEYWORDS = ['python', 'java', 'c++', 'sql', 'javascript', 'react', 'node', 'machine learning', 'nlp', 'data analysis']
DEGREE_KEYWORDS = ['bachelor', 'master', 'phd', 'b.sc', 'm.sc', 'b.e', 'b.tech', 'mba']
UNIVERSITY_KEYWORDS = ['university', 'institute', 'college', 'school']
EXPERIENCE_KEYWORDS = ['engineer', 'developer', 'manager', 'intern', 'consultant', 'designer', 'analyst']

# Text extraction

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file type")

# Info extraction

def preprocess_text(text):
    return [line.strip() for line in text.split('\n') if line.strip()]

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group() if match else None

def extract_phone(text):
    match = re.search(r"\+?\d[\d\s\-()]{8,}\d", text)
    return match.group() if match else None

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return extract_name_nltk(text)

def extract_name_nltk(text):
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(words)
        chunks = nltk.ne_chunk(tagged)
        for chunk in chunks:
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                return ' '.join(c[0] for c in chunk)
    return None

def extract_skills(text):
    return list({skill for skill in SKILL_KEYWORDS if re.search(rf'\b{re.escape(skill)}\b', text.lower())})

def is_education(line):
    return any(deg in line.lower() for deg in DEGREE_KEYWORDS + UNIVERSITY_KEYWORDS)

def is_experience(line):
    return any(role in line.lower() for role in EXPERIENCE_KEYWORDS)

def extract_edu_and_exp_with_dates(lines):
    education, experience = [], []
    for line in lines:
        date_matches = re.findall(r"(?i)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\.?\s?\d{4})\s*[-\u2013\u2014]\s*(Present|\d{4})", line)
        date_str = ' - '.join([' - '.join(match) for match in date_matches]) if date_matches else None
        if is_education(line):
            education.append({"text": line, "date": date_str})
        elif is_experience(line):
            experience.append({"text": line, "date": date_str})
    return education, experience

# Database

def store_in_database(data, db_path="resumes.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT, phone TEXT,
        skills TEXT, education TEXT, experience TEXT
    )''')

    cursor.execute('''INSERT INTO resumes (name, email, phone, skills, education, experience) VALUES (?, ?, ?, ?, ?, ?)''',
                   (
                       data["Name"],
                       data["Email"],
                       data["Phone"],
                       ", ".join(data["Skills"]),
                       "; ".join([e["text"] for e in data["Education"]]),
                       "; ".join([e["text"] for e in data["Experience"]])
                   ))
    conn.commit()
    conn.close()

# Main
if __name__ == "__main__":
    resume_file = "resume.pdf"  # path to file
    text = extract_text(resume_file)
    lines = preprocess_text(text)

    parsed = {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Phone": extract_phone(text),
        "Skills": extract_skills(text),
        "Education": [],
        "Experience": [],
    }

    parsed["Education"], parsed["Experience"] = extract_edu_and_exp_with_dates(lines)
    print("\n✅ Extracted Data:")
    for k, v in parsed.items():
        print(f"{k}: {v}")

    store_in_database(parsed)
    print("\n📁 Stored in database: resumes.db")