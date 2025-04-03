import os
import fitz  # PyMuPDF for extracting text from PDFs
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load SBERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load the JD ID Mapping from CSV
csv_path = "pdf_list.csv"  # Ensure this file is in the same directory
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    jd_mapping = dict(zip(df["ID"], df["JD"]))  # Map IDs to job names
else:
    jd_mapping = {}

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file."""
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text("text") for page in doc)
    return text.strip()

# Find Top N JD Matches
def get_top_jds(resume_pdf, jd_folder, top_n=5):
    """Matches resume with job descriptions and returns the top N results."""
    resume_text = extract_text_from_pdf(resume_pdf)
    jd_texts, jd_files = [], []

    # Match only against files listed in the CSV
    for file_id in jd_mapping.keys():
        file_path = os.path.join(jd_folder, file_id)
        if os.path.exists(file_path):
            jd_texts.append(extract_text_from_pdf(file_path))
            jd_files.append(file_id)

    # Compute SBERT embeddings
    resume_embedding = model.encode([resume_text], convert_to_numpy=True)
    jd_embeddings = model.encode(jd_texts, convert_to_numpy=True)

    # Compute cosine similarity
    similarities = cosine_similarity(resume_embedding, jd_embeddings)[0]

    # Sort results by similarity score
    scores = sorted(zip(jd_files, similarities), key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# Streamlit UI
st.title("📄 Find Jobs")

uploaded_resume = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])
jd_folder = "JD_db"  # Folder containing job descriptions

if uploaded_resume:
    if not jd_mapping:
        st.error("⚠️ No job descriptions found! Please upload 'pdf_list.csv' and ensure JD files exist.")
    else:
        top_n = st.slider("🔢 Select Number of Job Matches:", 1, min(10, len(jd_mapping)), 5)

        temp_resume_path = "temp_resume.pdf"
        with open(temp_resume_path, "wb") as f:
            f.write(uploaded_resume.getbuffer())

        matches = get_top_jds(temp_resume_path, jd_folder, top_n)

        st.write("### ✅ Top Matching Job Descriptions:")
        for i, (file_id, score) in enumerate(matches, 1):
            job_name = jd_mapping.get(file_id, "Unknown Job")  # Get job name from CSV
            file_path = os.path.join(jd_folder, file_id)

            with open(file_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"{i}. 📄 **{job_name}** - 🏆 Score: {round(score * 100, 2)}%")
            with col2:
                st.download_button(f"⬇️ Download", pdf_data, file_name=file_id, mime="application/pdf")
            with col3:
                apply_link = f"http://localhost:5173/projects/{file_id.replace('.pdf', '')}"  # Example job application link
                st.markdown(f"[🚀 Apply]( {apply_link} )", unsafe_allow_html=True)

        os.remove(temp_resume_path)
