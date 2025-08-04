# Same imports as before...
import os, re, cv2, fitz, spacy, pytesseract, nltk
from PIL import Image
from spacy.training import Example
from spacy.scorer import Scorer
from collections import defaultdict

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

nlp = spacy.load("custom_ner_model")

# Same keyword lists...
# Define keyword lists used for rule-based detection
SKILL_KEYWORDS = [
    'python', 'java', 'c++', 'sql', 'javascript', 'react', 'node',
    'machine learning', 'nlp', 'data analysis'
]

DEGREE_KEYWORDS = [
    'bachelor', 'master', 'phd', 'b.sc', 'm.sc', 'b.e', 'b.tech', 'mba'
]

UNIVERSITY_KEYWORDS = [
    'university', 'institute', 'college', 'school'
]

EXPERIENCE_KEYWORDS = [
    'engineer', 'developer', 'manager', 'intern',
    'consultant', 'designer', 'analyst'
]

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
        date_matches = re.findall(r"(?i)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\.?\s?\d{4})\s*[-–—]\s*(Present|\d{4})", line)
        date_str = ' - '.join([' - '.join(match) for match in date_matches]) if date_matches else None
        if is_education(line):
            education.append({"text": line, "date": date_str})
        elif is_experience(line):
            experience.append({"text": line, "date": date_str})
    return education, experience

def get_entity_spans(text, entity_list, label):
    spans = []
    for ent_text in entity_list:
        for match in re.finditer(re.escape(ent_text), text, flags=re.IGNORECASE):
            spans.append((match.start(), match.end(), label))
    return spans

def remove_overlapping_spans(spans):
    spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    result, prev_end = [], -1
    for start, end, label in spans:
        if start >= prev_end:
            result.append((start, end, label))
            prev_end = end
    return result

def fix_spans(text, entities, nlp):
    doc = nlp.make_doc(text)
    fixed = []
    for start, end, label in entities:
        token_start = None
        token_end = None
        for token in doc:
            if token.idx <= start < token.idx + len(token):
                token_start = token.i
            if token.idx < end <= token.idx + len(token):
                token_end = token.i + 1
        if token_start is None or token_end is None:
            print(f"⚠️ Misaligned: '{text[start:end]}'")
            continue
        fixed.append((doc[token_start].idx, doc[token_end - 1].idx + len(doc[token_end - 1]), label))
    return fixed

def parse_resume(file_path):
    text = extract_text(file_path)
    lines = preprocess_text(text)
    education, experience = extract_edu_and_exp_with_dates(lines)
    return {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Phone": extract_phone(text),
        "Skills": extract_skills(text),
        "Education": education,
        "Experience": experience,
        "RawText": text
    }

def parse_resume_with_spans(file_path):
    parsed = parse_resume(file_path)
    text = parsed["RawText"]
    entities = []

    if parsed["Name"]:
        entities.extend(get_entity_spans(text, [parsed["Name"]], "PERSON"))
    if parsed["Email"]:
        entities.extend(get_entity_spans(text, [parsed["Email"]], "EMAIL"))
    if parsed["Phone"]:
        entities.extend(get_entity_spans(text, [parsed["Phone"]], "PHONE"))
    if parsed["Skills"]:
        entities.extend(get_entity_spans(text, parsed["Skills"], "SKILL"))
    entities.extend(get_entity_spans(text, [e["text"] for e in parsed["Education"]], "EDUCATION"))
    entities.extend(get_entity_spans(text, [e["text"] for e in parsed["Experience"]], "EXPERIENCE"))

    entities = fix_spans(text, entities, nlp)
    entities = remove_overlapping_spans(entities)
    return (text, {"entities": entities})

def evaluate_ner_model(nlp, examples):
    scorer = Scorer()
    spacy_examples = []
    for text, annotations in examples:
        doc = nlp(text)
        example = Example.from_dict(doc, annotations)
        spacy_examples.append(example)
    scores = scorer.score(spacy_examples)
    return scores

if __name__ == "__main__":
    path = "resume.pdf"  # Change to your actual file
    parsed = parse_resume(path)

    print("\n🔍 Parsed Resume Info:")
    for k, v in parsed.items():
        if k == "RawText": continue
        print(f"{k}: {v}")

    text, annotations = parse_resume_with_spans(path)
    print("\n📏 Evaluating model...")
    scores = evaluate_ner_model(nlp, [(text, annotations)])
    print(f"📊 Precision: {scores['ents_p']:.3f}")
    print(f"📊 Recall:    {scores['ents_r']:.3f}")
    print(f"📊 F-score:   {scores['ents_f']:.3f}")
