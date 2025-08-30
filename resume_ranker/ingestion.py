import os
import zipfile
import tempfile
import shutil
from typing import List, Tuple

from PyPDF2 import PdfReader

__all__ = [
    "extract_pdfs_from_zip",
    "extract_text_from_pdf",
    "cleanup_temp_dir",
]

def extract_pdfs_from_zip(uploaded_zip_file) -> Tuple[str, List[str]]:
    """Save the uploaded ZIP to a temporary directory, extract PDFs and return the temp dir path and list of PDF file paths."""
    temp_dir = tempfile.mkdtemp(prefix="resumes_")
    zip_path = os.path.join(temp_dir, "resumes.zip")

    # Save zip to disk
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip_file.getbuffer())

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Collect pdfs
    pdf_files: List[str] = []
    for root, _dirs, files in os.walk(temp_dir):
        if "__MACOSX" in root:
            continue  # skip mac metadata
        for file in files:
            if file.startswith(".") or file.startswith("._"):
                continue
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    return temp_dir, pdf_files

def extract_text_from_pdf(pdf_file_path: str) -> str:
    """Extract text from a single PDF file."""
    try:
        if "__MACOSX" in pdf_file_path or os.path.basename(pdf_file_path).startswith("._"):
            return ""
        with open(pdf_file_path, "rb") as file:
            reader = PdfReader(file)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
    except Exception as e:
        # Return empty string on failure; caller can handle logging/alerts
        return ""

def cleanup_temp_dir(temp_dir: str):
    """Remove temporary directory created during processing."""
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)