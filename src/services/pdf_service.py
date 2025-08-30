import os
import tempfile
import zipfile
import shutil
from typing import List, Dict, Optional
import streamlit as st
from PyPDF2 import PdfReader
import fitz  # PyMuPDF for better text extraction
import logging

class PDFService:
    """Enhanced PDF processing service with multiple extraction methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str, method: str = "auto") -> str:
        """Extract text from PDF using multiple methods for best results"""
        
        # Skip macOS metadata files
        if "__MACOSX" in pdf_path or os.path.basename(pdf_path).startswith('._'):
            return ""
        
        text = ""
        
        if method == "auto":
            # Try PyMuPDF first (better for complex layouts)
            text = self._extract_with_pymupdf(pdf_path)
            
            # Fall back to PyPDF2 if PyMuPDF fails or returns poor results
            if len(text.strip()) < 100:
                text = self._extract_with_pypdf2(pdf_path)
        elif method == "pymupdf":
            text = self._extract_with_pymupdf(pdf_path)
        elif method == "pypdf2":
            text = self._extract_with_pypdf2(pdf_path)
        
        return self._clean_text(text)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (better for complex layouts)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            self.logger.warning(f"PyMuPDF extraction failed for {pdf_path}: {str(e)}")
            return ""
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            self.logger.warning(f"PyPDF2 extraction failed for {pdf_path}: {str(e)}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                cleaned_lines.append(line)
        
        # Join with single newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive spaces
        import re
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        
        return cleaned_text
    
    def process_zip_file(self, zip_file_buffer: bytes, temp_dir: str) -> List[Dict[str, str]]:
        """Process uploaded zip file and extract all PDF texts"""
        
        # Save zip file
        zip_path = os.path.join(temp_dir, "resumes.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file_buffer)
        
        # Extract zip file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                self.logger.info(f"Extracted {len(zip_ref.namelist())} files")
        except zipfile.BadZipFile:
            raise ValueError("Invalid ZIP file")
        except Exception as e:
            raise ValueError(f"Error extracting ZIP: {str(e)}")
        
        # Find and process PDF files
        pdf_files = self._find_pdf_files(temp_dir)
        
        if not pdf_files:
            raise ValueError("No PDF files found in ZIP")
        
        # Extract text from all PDFs
        resume_texts = []
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            text = self.extract_text_from_pdf(pdf_path)
            
            if text.strip():  # Only include non-empty texts
                resume_texts.append({
                    'filename': filename,
                    'text': text,
                    'file_path': pdf_path
                })
            else:
                self.logger.warning(f"No text extracted from {filename}")
        
        return resume_texts
    
    def _find_pdf_files(self, directory: str) -> List[str]:
        """Find all PDF files in directory, ignoring system files"""
        pdf_files = []
        
        for root, dirs, files in os.walk(directory):
            # Skip macOS metadata directories
            if "__MACOSX" in root:
                continue
            
            for file in files:
                # Skip hidden and metadata files
                if file.startswith('.') or file.startswith('._'):
                    continue
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        return pdf_files
    
    def validate_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Validate PDF file and return metadata"""
        try:
            # Try to open with PyMuPDF
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            
            # Get metadata
            metadata = doc.metadata
            doc.close()
            
            return {
                'valid': True,
                'page_count': page_count,
                'metadata': metadata,
                'file_size': os.path.getsize(pdf_path)
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }