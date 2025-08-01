import re
import pytesseract
import spacy
from PIL import Image
import fitz  # PyMuPDF
import cv2
import os

nlp = spacy.load("en_core_web_sm")


SKILL_KEYWORDS = ['python', 'java', 'c++', 'sql', 'javascript', 'react', 'node', 'machine learning', 'nlp', 'data analysis']
DEGREE_KEYWORDS = ['bachelor', 'master', 'phd', 'b.sc', 'm.sc', 'b.e', 'b.tech', 'mba']
UNIVERSITY_KEYWORDS = ['university', 'institute', 'college', 'school']
EXPERIENCE_KEYWORDS = ['engineer', 'developer', 'manager', 'intern', 'consultant', 'designer', 'analyst']


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
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
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines



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
    return None

def extract_skills(text):
    text = text.lower()
    return list(set(skill for skill in SKILL_KEYWORDS if skill in text))



def extract_dates(text):
    # Match date ranges: "Jan 2020 - Dec 2022", "2019 – Present", "2015 - 2019"
    pattern = r"(?i)(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\.?\s?\d{4})\s*[-–—]\s*(?:Present|\d{4})"
    return re.findall(pattern, text)

def is_education(line):
    line = line.lower()
    return any(deg in line for deg in DEGREE_KEYWORDS + UNIVERSITY_KEYWORDS)

def is_experience(line):
    line = line.lower()
    return any(role in line for role in EXPERIENCE_KEYWORDS)

def extract_edu_and_exp_with_dates(lines):
    education = []
    experience = []

    for line in lines:
        date_matches = re.findall(r"(?i)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\.?\s?\d{4})\s*[-–—]\s*(Present|\d{4})", line)
        date_str = ' - '.join([' - '.join(match) for match in date_matches]) if date_matches else None

        clean_line = line.strip()

        if is_education(clean_line):
            education.append({"text": clean_line, "date": date_str})
        elif is_experience(clean_line):
            experience.append({"text": clean_line, "date": date_str})

    return education, experience



def parse_resume(file_path):
    raw_text = extract_text(file_path)
    lines = preprocess_text(raw_text)

    education, experience = extract_edu_and_exp_with_dates(lines)

    return {
        "Name": extract_name(raw_text),
        "Email": extract_email(raw_text),
        "Phone": extract_phone(raw_text),
        "Skills": extract_skills(raw_text),
        "Education": education,
        "Experience": experience
    }


if __name__ == "__main__":
    file_path = "resume.pdf"
    result = parse_resume(file_path)
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
