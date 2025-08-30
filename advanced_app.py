import os
import time
import json
from typing import List, Dict

import streamlit as st
import pandas as pd

from resume_ranker.ingestion import extract_pdfs_from_zip, extract_text_from_pdf, cleanup_temp_dir
from resume_ranker.ranking import ResumeRanker

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Advanced Resume Ranker", page_icon="📄", layout="wide")

st.title("Advanced Resume Ranker 📄")

# --- Sidebar settings
with st.sidebar:
    st.header("Configuration")
    model = st.selectbox("LLM Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    embedding_model = st.selectbox("Embedding Model", ["text-embedding-3-small", "text-embedding-3-large"])
    top_k = st.slider("Candidates to retrieve", 5, 50, value=15)
    weight_llm = st.slider("Weight LLM Score (%)", 0, 100, 70)

    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY not found in environment. Set it before proceeding.")

# --- Main
uploaded_zip = st.file_uploader("Upload ZIP of resumes (PDF)", type=["zip"])
query_text = st.text_area("Enter Job Description / Ranking Criteria", "Looking for Python developer with ML experience and CS degree.")

if uploaded_zip and st.button("Rank Resumes"):
    if not OPENAI_API_KEY:
        st.error("Please set OPENAI_API_KEY environment variable.")
        st.stop()

    temp_dir, pdf_files = extract_pdfs_from_zip(uploaded_zip)

    if not pdf_files:
        st.error("No PDF resumes found in the uploaded file.")
        cleanup_temp_dir(temp_dir)
        st.stop()

    ranker = ResumeRanker(openai_api_key=OPENAI_API_KEY, model=model, embedding_model=embedding_model)

    texts: List[str] = []
    metas: List[Dict] = []

    progress = st.progress(0.0)
    status = st.empty()

    for idx, pdf in enumerate(pdf_files):
        status.text(f"Extracting text from {os.path.basename(pdf)} ({idx+1}/{len(pdf_files)})")
        text = extract_text_from_pdf(pdf)
        texts.append(text)
        metas.append({"filename": os.path.basename(pdf), "resume_text": text})
        progress.progress((idx + 1) / len(pdf_files) * 0.4)

    ranker.add_resumes(texts, metas)

    status.text("Ranking resumes with AI ...")
    results = ranker.rank(query_text, top_k=top_k)

    # Build DataFrame
    df = pd.DataFrame([
        {
            "Rank": i + 1,
            "File": r["filename"],
            "LLM Score": r.get("match_score", 0),
            "Similarity": round(r.get("similarity", 0) * 100, 2),
            "Combined": round(0.7 * r.get("match_score", 0) + 0.3 * r.get("similarity", 0) * 100, 2),
        }
        for i, r in enumerate(results)
    ])

    st.subheader("Ranking Results")
    st.dataframe(df, use_container_width=True)

    # Detailed view
    st.subheader("Detailed Analysis")
    for i, r in enumerate(results):
        with st.expander(f"{i+1}. {r['filename']} - LLM Score: {r.get('match_score', 0)}"):
            st.markdown("**Reasoning:**")
            st.markdown(r.get("reasoning", "N/A"))
            st.markdown("**Resume Text (first 1000 chars):**")
            st.code(r.get("resume_text", "")[:1000])

    # Download JSON
    json_str = json.dumps(results, indent=2)
    st.download_button("Download Raw JSON", json_str, file_name="resume_ranking.json", mime="application/json")

    cleanup_temp_dir(temp_dir)
    progress.empty()
    status.text("Done.")