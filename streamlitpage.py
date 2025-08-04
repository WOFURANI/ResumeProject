import streamlit as st
import spacy
import re
import pandas as pd
from io import StringIO
import pdfplumber
import subprocess
import importlib.util
# Load SpaCy model
try:
    nlp = spacy.load("custom_ner_model")  # Try custom model
except:
    nlp = spacy.load("en_core_web_sm")

# Keywords
skills = ["javascript", "react", "docker", "laravel", "typescript", "python", "java", "css"]
experience_keywords = ["engineer", "developer", "intern", "manager", "consultant"]
education_keywords = ["university", "institute", "college", "school", "education", "bachelor", "master", "phd"]

# ----------------------------
# Extract structured entities
# ----------------------------
def extract_info(text):
    text_lower = text.lower()
    doc = nlp(text)

    data = {
        "Name": "",
        "Email": "",
        "Phone": "",
        "Experience": [],
        "Skills": [],
        "Education": []
    }

    # Name
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            data["Name"] = ent.text
            break

    # Email
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    if match:
        data["Email"] = match.group()

    # Phone
    match = re.search(r"(\+?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4})", text)
    if match:
        data["Phone"] = match.group()

    # Experience
    for kw in experience_keywords:
        if kw in text_lower:
            data["Experience"].append(kw)

    # Skills
    for kw in skills:
        if kw in text_lower:
            data["Skills"].append(kw)

    # Education
    for kw in education_keywords:
        if kw in text_lower:
            data["Education"].append(kw)

    return data

def load_spacy_model(model_name="en_core_web_sm"):
    if importlib.util.find_spec(model_name) is None:
        subprocess.run(["python", "-m", "spacy", "download", model_name])
    return spacy.load(model_name)

# Load the model
nlp = load_spacy_model()
# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📄 Resume Parser App")

uploaded_file = st.file_uploader("Upload a resume (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = stringio.read()

    # Extract entities
    result = extract_info(text)

    st.subheader("✅ Extracted Information")
    st.json(result)

    # Save to CSV
    if st.button("💾 Save to CSV"):
        df = pd.DataFrame([result])
        df["Experience"] = df["Experience"].apply(lambda x: "; ".join(x))
        df["Skills"] = df["Skills"].apply(lambda x: "; ".join(x))
        df["Education"] = df["Education"].apply(lambda x: "; ".join(x))
        df.to_csv("parsed_resume.csv", index=False)
        st.success("Saved to parsed_resume.csv ✅")
