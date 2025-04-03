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

# Load the Resume ID Mapping from CSV
res_csv_path = "res_list.csv"  # Ensure this file is in the same directory
if os.path.exists(res_csv_path):
    res_df = pd.read_csv(res_csv_path)
    resume_mapping = dict(zip(res_df["ID"], res_df["Name"]))  # Map IDs to candidate names
else:
    resume_mapping = {}

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file."""
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text("text") for page in doc)
    return text.strip()

# Find Top N Resume Matches
def get_top_resumes(jd_pdf, resume_folder, top_n=5):
    """Matches JD with resumes and returns the top N results."""
    jd_text = extract_text_from_pdf(jd_pdf)
    resume_texts, resume_files = [], []

    # Match only against files listed in the CSV
    for file_id in resume_mapping.keys():
        file_path = os.path.join(resume_folder, file_id)
        if os.path.exists(file_path):
            resume_texts.append(extract_text_from_pdf(file_path))
            resume_files.append(file_id)

    # Compute SBERT embeddings
    jd_embedding = model.encode([jd_text], convert_to_numpy=True)
    resume_embeddings = model.encode(resume_texts, convert_to_numpy=True)

    # Compute cosine similarity
    similarities = cosine_similarity(jd_embedding, resume_embeddings)[0]

    # Sort results by similarity score
    scores = sorted(zip(resume_files, similarities), key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# Streamlit UI
st.title("📑 Find Candidates")

uploaded_jd = st.file_uploader("📤 Upload Job Description (PDF)", type=["pdf"])
resume_folder = "resume_db"  # Folder containing resumes

if uploaded_jd:
    if not resume_mapping:
        st.error("⚠️ No resumes found! Please upload 'res_list.csv' and ensure resume files exist.")
    else:
        top_n = st.slider("🔢 Select Number of Resume Matches:", 1, min(10, len(resume_mapping)), 5)

        temp_jd_path = "temp_jd.pdf"
        with open(temp_jd_path, "wb") as f:
            f.write(uploaded_jd.getbuffer())

        matches = get_top_resumes(temp_jd_path, resume_folder, top_n)

        st.write("### ✅ Top Matching Candidates:")
        for i, (file_id, score) in enumerate(matches, 1):
            candidate_name = resume_mapping.get(file_id, "Unknown Candidate")  # Get candidate name from CSV
            file_path = os.path.join(resume_folder, file_id)

            with open(file_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"{i}. 👤 **{candidate_name}** - 🏆 Score: {round(score * 100, 2)}%")
            with col2:
                st.download_button(f"⬇️ Download", pdf_data, file_name=file_id, mime="application/pdf")
            with col3:
                contact_link = "http://localhost:5173/chats"  # Default chat link
                st.markdown(f"[📩 Contact]( {contact_link} )", unsafe_allow_html=True)

        os.remove(temp_jd_path)
