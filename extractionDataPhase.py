import os
import re
import cv2
import fitz  # PyMuPDF
import spacy
import pytesseract
import nltk
from PIL import Image
from spacy.training import Example
from spacy.scorer import Scorer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load your trained model
nlp = spacy.load("custom_ner_model")

# Keywords
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

def get_entity_spans(text, entity_list, label):
    spans = []
    for ent_text in entity_list:
        start = text.find(ent_text)
        if start != -1:
            end = start + len(ent_text)
            spans.append((start, end, label))
    return spans

def remove_overlapping_spans(spans):
    spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    filtered_spans = []
    prev_end = -1
    for span in spans:
        start, end, label = span
        if start >= prev_end:
            filtered_spans.append(span)
            prev_end = end
    return filtered_spans

def fix_spans(text, entities, nlp):
    doc = nlp.make_doc(text)
    fixed_entities = []
    for start, end, label in entities:
        token_start = None
        token_end = None
        for token in doc:
            if token.idx <= start < token.idx + len(token):
                token_start = token.idx
            if token.idx < end <= token.idx + len(token):
                token_end = token.idx + len(token)
        if token_start is None or token_end is None:
            print(f"Warning: skipping entity '{text[start:end]}' due to bad alignment")
            continue
        fixed_entities.append((token_start, token_end, label))
    return fixed_entities

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
        "Experience": experience,
        "RawText": raw_text
    }

def parse_resume_with_spans(file_path, use_nltk_for_name=False):
    parsed = parse_resume(file_path, use_nltk_for_name)
    text = parsed["RawText"]

    entities = []

    if parsed["Name"]:
        entities.extend(get_entity_spans(text, [parsed["Name"]], "PERSON"))
    if parsed["Email"]:
        entities.extend(get_entity_spans(text, [parsed["Email"]], "EMAIL"))
    if parsed["Phone"]:
        entities.extend(get_entity_spans(text, [parsed["Phone"]], "PHONE"))
    if parsed["Skills"]:
        entities.extend(get_entity_spans(text.lower(), parsed["Skills"], "SKILL"))
    edu_texts = [edu["text"] for edu in parsed["Education"]]
    entities.extend(get_entity_spans(text.lower(), [e.lower() for e in edu_texts], "EDUCATION"))
    exp_texts = [exp["text"] for exp in parsed["Experience"]]
    entities.extend(get_entity_spans(text.lower(), [e.lower() for e in exp_texts], "EXPERIENCE"))

    # Fix entity spans to token boundaries
    entities = fix_spans(text, entities, nlp)
    # Remove overlaps
    entities = remove_overlapping_spans(entities)

    return (text, {"entities": entities})

def evaluate_ner_model(nlp, examples):
    scorer = Scorer()
    spacy_examples = []

    for text, annotations in examples:
        print("\nEvaluating text:\n", text)
        print("True entities:", annotations["entities"])

        doc = nlp(text)
        print("Predicted entities:", [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])

        example = Example.from_dict(doc, annotations)
        spacy_examples.append(example)

    scores = scorer.score(spacy_examples)
    return scores


if __name__ == "__main__":
    file_path = "resume.pdf"  # Change to your test file path

    # Parse resume normally
    result = parse_resume(file_path, use_nltk_for_name=False)
    print("\nParsed Resume Data:")
    for key, value in result.items():
        if key == "RawText":
            continue
        print(f"{key}:")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    print(f"  - {item['text']} ({item['date']})" if item['date'] else f"  - {item['text']}")
                else:
                    print(f"  - {item}")
        else:
            print(f"  {value}")

    # Prepare evaluation data with spans for this resume
    eval_data = [parse_resume_with_spans(file_path)]

    # Evaluate the custom NER model on this parsed data
    results = evaluate_ner_model(nlp, eval_data)
    print("\nNER Model Evaluation Metrics:")
    print(f"Precision: {results['ents_p']:.3f}")
    print(f"Recall: {results['ents_r']:.3f}")
    print(f"F-score: {results['ents_f']:.3f}")
