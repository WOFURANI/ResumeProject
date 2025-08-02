import os
import re
import cv2
import fitz  # PyMuPDF
import spacy
import pytesseract
import nltk
from PIL import Image

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

nlp = spacy.load("en_core_web_sm")

# Define keywords
SKILL_KEYWORDS = ['python', 'java', 'c++', 'sql', 'javascript', 'react', 'node',
                  'machine learning', 'nlp', 'data analysis']
DEGREE_KEYWORDS = ['bachelor', 'master', 'phd', 'b.sc', 'm.sc', 'b.e', 'b.tech', 'mba']
UNIVERSITY_KEYWORDS = ['university', 'institute', 'college', 'school']
EXPERIENCE_KEYWORDS = ['engineer', 'developer', 'manager', 'intern', 'consultant', 'designer', 'analyst']


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


def preprocess_text(text):
    return [line.strip() for line in text.split('\n') if line.strip()]


def extract_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group() if match else None

def extract_phone(text):
    match = re.search(r"\+?\d[\d\s\-()]{8,}\d", text)
    return match.group() if match else None

def extract_name_spacy(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

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
    text = text.lower()
    return list(set(skill for skill in SKILL_KEYWORDS if skill in text))

def extract_dates(text):
    pattern = r"(?i)(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\.?\s?\d{4})\s*[-–—]\s*(?:Present|\d{4})"
    return re.findall(pattern, text)

def is_education(line):
    line = line.lower()
    return any(deg in line for deg in DEGREE_KEYWORDS + UNIVERSITY_KEYWORDS)

def is_experience(line):
    line = line.lower()
    return any(role in line for role in EXPERIENCE_KEYWORDS)

def extract_edu_and_exp_with_dates(lines):
    education, experience = [], []
    for line in lines:
        date_matches = re.findall(
            r"(?i)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\.?\s?\d{4})\s*[-–—]\s*(Present|\d{4})", line)
        date_str = ' - '.join([' - '.join(match) for match in date_matches]) if date_matches else None
        if is_education(line):
            education.append({"text": line, "date": date_str})
        elif is_experience(line):
            experience.append({"text": line, "date": date_str})
    return education, experience


# Resume Parsing
def parse_resume(file_path, use_nltk_for_name=False):
    raw_text = extract_text(file_path)
    lines = preprocess_text(raw_text)

    education, experience = extract_edu_and_exp_with_dates(lines)

    return {
        "Name": extract_name_nltk(raw_text) if use_nltk_for_name else extract_name_spacy(raw_text),
        "Email": extract_email(raw_text),
        "Phone": extract_phone(raw_text),
        "Skills": extract_skills(raw_text),
        "Education": education,
        "Experience": experience
    }
if __name__ == "__main__":
    file_path = "resume.pdf"  # Change to your test file
    result = parse_resume(file_path, use_nltk_for_name=False)

    for key, value in result.items():
        print(f"{key}:")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    print(f"  - {item['text']} ({item['date']})" if item['date'] else f"  - {item['text']}")
                else:
                    print(f"  - {item}")
        else:
            print(f"  {value}")
