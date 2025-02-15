import streamlit as st
import pdfplumber
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

# Load NLP model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from resumes
def extract_text(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except:
        text = "Error extracting text."
    return text

# Function to clean text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Function to compute similarity score
def compute_match_score(job_desc, resume_text):
    job_embedding = model.encode(job_desc, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(job_embedding, resume_embedding).item()
    return round(similarity * 100, 2)

# UI setup
st.set_page_config(page_title="Resume Screener", layout="wide")

# Custom CSS for futuristic dark theme
st.markdown(
    """
    <style>
        body {
            background-color: rgb(0, 0, 0);
            color: rgb(255, 255, 255);
            font-family: 'Arial', sans-serif;
        }
        .stTextArea, .stFileUploader, .stProgress {
            background-color: rgb(1, 1, 1);
            border-radius: 10px;
            padding: 14px;
            color: white !important;
        }
        h1, h2, h3, h4 {
            color: rgb(255, 255, 255);
        }
        .stProgress > div > div {
            background-color: rgb(0, 0, 0) !important;
        }
        label {
            color: rgb(255, 255, 255) !important;
            font-size: 16px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("C:\\Users\\marta\\Desktop\\CV\\logo.png", width=600)

st.subheader("Job Description")
job_desc = st.text_area("Paste Job Description", height=200)

st.subheader("Upload Resumes")
uploaded_files = st.file_uploader("Upload resumes (PDF only)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    results = []
    
    for file in uploaded_files:
        resume_text = extract_text(file)
        clean_resume_text = preprocess_text(resume_text)
        match_score = compute_match_score(job_desc, clean_resume_text)
        results.append((file.name, match_score))

    # Sort by match score
    results.sort(key=lambda x: x[1], reverse=True)

    # Display results
    st.subheader("Ranked Resumes")
    for i, (filename, score) in enumerate(results):
        st.markdown(f"<h3>{i+1}. {filename} - Match Score: {score}%</h3>", unsafe_allow_html=True)
        st.progress(score / 100)