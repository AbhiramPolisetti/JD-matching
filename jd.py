import os
import fitz  # PyMuPDF for extracting text from PDFs
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load SBERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file."""
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text("text") for page in doc)
    return text.strip()

# Find Top N Resume Matches
def get_top_resumes(jd_pdf, resume_folder, top_n=5):
    """Matches job description with resumes and returns the top N results."""
    jd_text = extract_text_from_pdf(jd_pdf)
    resume_texts, resume_files = [], []
    
    for file in os.listdir(resume_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(resume_folder, file)
            resume_texts.append(extract_text_from_pdf(file_path))
            resume_files.append(file)

    # Compute SBERT embeddings
    jd_embedding = model.encode([jd_text], convert_to_numpy=True)
    resume_embeddings = model.encode(resume_texts, convert_to_numpy=True)

    # Compute cosine similarity
    similarities = cosine_similarity(jd_embedding, resume_embeddings)[0]

    # Sort results by similarity score
    scores = sorted(zip(resume_files, similarities), key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# Streamlit UI
st.title("📝 Job Description → Resume Matching")

uploaded_jd = st.file_uploader("📤 Upload Job Description (PDF)", type=["pdf"])
resume_folder = "resume_db"  # Folder containing resumes

if uploaded_jd:
    if not os.listdir(resume_folder):
        st.error("⚠️ No resumes found! Please add some PDFs to the 'resume_db' folder.")
    else:
        top_n = st.slider("🔢 Select Number of Resume Matches:", 1, min(10, len(os.listdir(resume_folder))), 5)

        temp_jd_path = "temp_jd.pdf"
        with open(temp_jd_path, "wb") as f:
            f.write(uploaded_jd.getbuffer())

        matches = get_top_resumes(temp_jd_path, resume_folder, top_n)

        st.write("### ✅ Top Matching Resumes:")
        for i, (file, score) in enumerate(matches, 1):
            file_path = os.path.join(resume_folder, file)

            with open(file_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"{i}. 📄 **{file}** - 🏆 Score: {round(score * 100, 2)}%")
            with col2:
                st.download_button(f"⬇️ Download", pdf_data, file_name=file, mime="application/pdf")
            with col3:
                contact_link = f"https://www.linkedin.com/in/{file.replace('.pdf', '')}"  
                st.markdown(f"[📩 Contact]( {contact_link} )", unsafe_allow_html=True)

        os.remove(temp_jd_path)
