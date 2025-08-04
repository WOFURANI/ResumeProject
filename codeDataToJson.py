import pandas as pd
import re
import json

# Load CSV
df = pd.read_csv("Resume.csv")
resumes = df["Resume_str"].dropna().tolist()

# Define patterns and keywords
skills = ["javascript", "react", "docker", "laravel", "typescript", "python", "java", "css"]
education_keywords = ["bachelor", "master", "degree", "university", "college"]
experience_keywords = ["experience", "worked", "managed", "led", "project", "developed"]
email_pattern = r'\b[\w.-]+?@\w+?\.\w+?\b'
phone_pattern = r'\b(?:\+?\d{1,3})?[ -.]?\(?\d{2,4}\)?[ -.]?\d{3,4}[ -.]?\d{4}\b'

# Clean text function
def clean_text(text):
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# Extract entities
def extract_entities(text):
    entities = []
    lowered = text.lower()

    for match in re.finditer(email_pattern, text):
        entities.append((match.start(), match.end(), "EMAIL"))
    for match in re.finditer(phone_pattern, text):
        entities.append((match.start(), match.end(), "PHONE"))
    for skill in skills:
        for match in re.finditer(r"\b{}\b".format(re.escape(skill)), lowered):
            entities.append((match.start(), match.end(), "SKILL"))
    for keyword in education_keywords:
        for match in re.finditer(r"\b{}\b".format(re.escape(keyword)), lowered):
            entities.append((match.start(), match.end(), "EDUCATION"))
    for keyword in experience_keywords:
        for match in re.finditer(r"\b{}\b".format(re.escape(keyword)), lowered):
            entities.append((match.start(), match.end(), "EXPERIENCE"))

    return sorted(entities, key=lambda x: x[0])

# Create annotated data
annotated = []
for raw_text in resumes:
    cleaned = clean_text(raw_text)
    entities = extract_entities(cleaned)
    annotated.append({
        "text": cleaned,
        "entities": entities
    })

# Export
with open("cleaned_annotated_resumes.json", "w", encoding="utf-8") as f:
    json.dump(annotated, f, indent=2)
