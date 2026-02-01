import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from pydantic import BaseModel, Field
from typing import List,Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pdfplumber
import matplotlib.pyplot as plt
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# -------------------------------
# SETUP
# -------------------------------
load_dotenv()

st.set_page_config(page_title="AI Resume Match Analyzer", layout="wide")
st.title("AI Resume–Job Description Match Analyzer")

# Cache model
@st.cache_resource
def load_model():
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3, max_retries=2)

model = load_model()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded resume PDF"""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

FAISS_INDEX_PATH = "resume_index.faiss"
RESUME_META_PATH = "resume_metadata.pkl"

def load_or_create_faiss(dimension=384):  # 384 dims for MiniLM
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(RESUME_META_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(RESUME_META_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(dimension)
        metadata = []
    return index, metadata

faiss_index, resume_metadata = load_or_create_faiss()

def add_resume_to_faiss(name, text):
    vector = embed_model.encode([text])
    faiss_index.add(np.array(vector, dtype=np.float32))
    resume_metadata.append({"name": name, "text": text})
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(RESUME_META_PATH, "wb") as f:
        pickle.dump(resume_metadata, f)

def clear_faiss_database():
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(RESUME_META_PATH):
        os.remove(RESUME_META_PATH)
    return faiss.IndexFlatL2(384), []



def calculate_match_percentage(job_skills, resume_skills):
    job_set = set(s.lower() for s in job_skills)
    resume_set = set(s.lower() for s in resume_skills)
    matched = job_set.intersection(resume_set)
    missing = job_set - matched

    if len(job_set) == 0:
        return 0.0, matched, missing

    percentage = (len(matched) / len(job_set)) * 100
    return round(percentage, 1), matched, missing

def retrieve_relevant_resumes(job_description, top_k=3):
    job_vec = embed_model.encode([job_description])
    distances, indices = faiss_index.search(np.array(job_vec, dtype=np.float32), top_k *2)
    seen = set()
    results = []
    for i in indices[0]:
        if i < len(resume_metadata) and i not in seen:
            seen.add(i)
            results.append(resume_metadata[i])
        if len(results) >= top_k:
            break
    return results

# -------------------------------
# Structured Output Schema
# -------------------------------
class ResumeAnalysis(BaseModel):
    job_skills: List[str] = Field(description="List of 5–10 key skills or technologies from the job description.")
    resume_skills: List[str] = Field(description="List of 5–10 key skills or technologies from the resume.")
    match_percentage: Optional[float] = Field(default=0.0, description="Percentage of job-relevant skills that appear in the resume.")
    strong_points: Optional[List[str]] = Field(default=[], description="Top 3 strengths or strong alignment points.")
    weak_points: Optional[List[str]] = Field(default=[], description="Top 3 missing or weak skills relative to the job.")
    improvement_suggestions: Optional[List[str]] = Field(default=[], description="3 ways the resume can be improved to better match the job.")


# -------------------------------
# STREAMLIT UI
# -------------------------------

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area(" Paste Job Description", height=250, placeholder="Paste job description here...")

with st.sidebar:
    st.header("Model Settings")
    st.caption("Adjust LLM behavior for finer control.")
    temp = st.slider("Model Creativity (temperature)", 0.0, 1.0, 0.3)
    st.divider()

    if st.button(" Clear Resumes from database"):
        faiss_index, resume_metadata = clear_faiss_database()


    if resume_file and st.button("Store Resume in Database"):
        text = extract_text_from_pdf(resume_file)
        name = resume_file.name
        add_resume_to_faiss(name, text)
        st.success(f"Stored {name} in resume database!")

    #  Show available resumes
    if resume_metadata:
        st.subheader("Stored Resumes")
        for i, resume in enumerate(resume_metadata, 1):
            st.write(f"{i}. {resume['name']}")
    else:
        st.info("No resumes stored yet.")

    st.divider()

if st.button("Analyze Match"):
    if not job_description.strip():
        st.warning("Please paste a job description.")
        st.stop()

    relevant_resumes = retrieve_relevant_resumes(job_description, top_k=3)

    if not relevant_resumes:
        st.warning("No resumes found in FAISS database. Please store some first.")
        st.stop()

    all_results = []

    # --- Loop through retrieved resumes ---
    for idx, resume in enumerate(relevant_resumes, 1):
        st.markdown(f"###  Resume {idx}: {resume['name']}")
        with st.spinner(f"Analyzing match for {resume['name']}... ⏳"):

            combined_input = f"""
            You are an expert career coach and resume analyzer.

            Compare the JOB DESCRIPTION with the RESUME and return ONLY valid JSON.
            Do NOT include markdown, code blocks, or extra text.

            JSON format:
            {{
                "job_skills": ["skill1", "skill2"],
                "resume_skills": ["skill1", "skill2"],
                "match_percentage": 0.0,
                "strong_points": ["point1", "point2", "point3"],
                "weak_points": ["point1", "point2", "point3"],
                "improvement_suggestions": ["suggestion1", "suggestion2", "suggestion3"]
            }}

            ### JOB DESCRIPTION:
            {job_description}

            ### RESUME:
            {resume['text']}
            """

            try:
                raw_result = model.invoke(combined_input)
                content = raw_result.content.strip()
                # Clean unwanted markdown
                content = content.replace("```json", "").replace("```", "").strip()
                # Extract JSON block only
                import re
                match = re.search(r"\{.*\}", content, re.S)
                if match:
                    content = match.group(0)

                parsed = json.loads(content)
                result = ResumeAnalysis(**parsed)

                # Calculate match % manually
                calc_match, matched, missing = calculate_match_percentage(
                    result.job_skills, result.resume_skills
                )
                result.match_percentage = calc_match
                 # Filter only resumes with > 40% match
                
                all_results.append({
                    "name": resume["name"],
                    "analysis": result,
                    "matched": matched,
                    "missing": missing
                })
            

            except Exception as e:
                st.error(f" Failed to process {resume['name']}: {e}")
                continue
            
    filtered_results = [r for r in all_results if r["analysis"].match_percentage >= 40]
    if filtered_results:
        for fr in filtered_results:
            # Inline quick summary for each
            result = fr["analysis"]
            st.success(f"{fr['name']} → {result.match_percentage:.1f}% match")

    # -------------------------------
    # AFTER analyzing all resumes
    # -------------------------------
    if not all_results:
        st.error("No valid analyses were produced.")
        st.stop()

    
    if not filtered_results:
        st.warning("⚠️ No resumes matched above 40%. Try adding more resumes or revising the job description.")
        st.stop()


    # Pick best match
    best_resume = max(filtered_results, key=lambda x: x["analysis"].match_percentage)
    result = best_resume["analysis"]

    st.markdown("---")
    st.success(f"🏆 **Best Match:** {best_resume['name']} — {result.match_percentage:.1f}%")

    # --- Summary metrics ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric(" Match Percentage", f"{result.match_percentage:.1f}%")
    with col2:
        st.metric(" Total Job Skills", len(result.job_skills))

    # --- Skills comparison ---
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Job Skills")
        st.write(", ".join(result.job_skills))
    with col4:
        st.subheader("Resume Skills")
        st.write(", ".join(result.resume_skills))

    st.markdown("---")

    # --- Strong & Weak points ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Strong Points")
        for s in result.strong_points:
            st.write(f" {s}")
    with col2:
        st.subheader(" Weak Points")
        for w in result.weak_points:
            st.write(f"{w}")

    st.markdown("---")

    # --- Charts & Visuals ---
    matched = [s for s in result.job_skills if s.lower() in [r.lower() for r in result.resume_skills]]
    missing = [s for s in result.job_skills if s.lower() not in [r.lower() for r in result.resume_skills]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(["Matched", "Missing"], [len(matched), len(missing)], color=["green", "red"])
    ax.set_xlabel("Skill Count")
    st.pyplot(fig)

    # --- Suggestions ---
    st.subheader(" Resume Improvement Suggestions")
    for i, suggestion in enumerate(result.improvement_suggestions, 1):
        st.write(f"{i}. {suggestion}")

    