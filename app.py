import streamlit as st
import zipfile
import os
import tempfile
import shutil
import io
import json
from PyPDF2 import PdfReader
import pandas as pd
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# Set page config
st.set_page_config(
    page_title="Resume Ranker (GPT-4o)",
    page_icon="📄",
    layout="wide"
)

# App title and description
st.title("Resume Ranker (Powered by GPT-4o)")
st.markdown("Upload a zip file containing resumes in PDF format, specify ranking criteria, and get AI-powered analysis and ranking.")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    try:
        # Skip macOS metadata files
        if "__MACOSX" in pdf_file_path or os.path.basename(pdf_file_path).startswith('._'):
            return ""
            
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
            return text
    except Exception as e:
        st.error(f"Error extracting text from {os.path.basename(pdf_file_path)}: {str(e)}")
        return ""

# Function to analyze resume with LangChain and OpenAI
def analyze_resume(resume_text, prompt, model="gpt-4o"):
    try:
        # Create LangChain ChatOpenAI instance
        llm = ChatOpenAI(
            model=model, 
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Create a system prompt that instructs the model
        system_template = """
        You are a resume analysis expert. Extract relevant information from the resume text according to the user's criteria.
        Provide a structured JSON response with the following fields:
        - candidate_name: The name of the candidate
        - relevant_skills: List of skills relevant to the criteria
        - relevant_experience: Summary of experience relevant to the criteria
        - education: Summary of education
        - match_score: A score from 0-100 indicating how well the resume matches the criteria
        - reasoning: Brief explanation for the score
        
        Ranking Criteria: {criteria}
        
        Resume Text:
        {resume_text}
        
        Based on the above resume and ranking criteria, provide the JSON analysis.
        """
        
        # Create prompt template
        prompt_template = PromptTemplate.from_template(system_template)
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            criteria=prompt,
            resume_text=resume_text[:8000]  # Limiting text to avoid token limits
        )
        
        # Send to LLM
        response = llm.invoke(formatted_prompt)
        result = response.content
        
        # Try to parse the JSON response
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # If response is not valid JSON, try to extract JSON part
            import re
            json_match = re.search(r'({.*})', result.replace('\n', ''), re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except:
                    st.warning("Could not parse JSON response. Returning raw text.")
                    return {"error": "JSON parsing failed", "raw_response": result}
            else:
                st.warning("Could not extract JSON from response. Returning raw text.")
                return {"error": "JSON extraction failed", "raw_response": result}
                
    except Exception as e:
        st.error(f"Error analyzing resume: {str(e)}")
        return {"error": str(e)}

# Main function
def main():
    # File upload
    uploaded_zip = st.file_uploader("Upload ZIP file containing resumes (PDF format)", type=["zip"], accept_multiple_files=False, key="resume_zip_uploader")
    
    # Display helpful instructions when no file is uploaded
    if uploaded_zip is None:
        st.info("Please upload a ZIP file containing PDF resumes. Make sure your ZIP file is not password protected.")
        st.markdown("""
        ### Tips for successful upload:
        - Make sure your ZIP file contains PDF resumes
        - File size should be under 200MB
        - If the upload button doesn't work, try refreshing the page
        - Check that your browser allows file uploads
        """)
    else:
        st.success(f"Successfully uploaded: {uploaded_zip.name}")

    # Ranking criteria (only show if a file is uploaded)
    if uploaded_zip is not None:
        ranking_criteria = st.text_area(
            "Enter your ranking criteria",
            "Looking for candidates with at least 3 years of experience in Python development, "
            "knowledge of machine learning frameworks, and a degree in Computer Science or related field."
        )
        
        if uploaded_zip is not None:
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Save the uploaded zip file
                zip_path = os.path.join(temp_dir, "resumes.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())
                
                st.info(f"ZIP file saved to temporary location. Size: {len(uploaded_zip.getbuffer())/1024:.2f} KB")
                
                # Extract the zip file
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                        st.info(f"ZIP file extracted successfully. Contents: {len(zip_ref.namelist())} files")
                except zipfile.BadZipFile:
                    st.error("The uploaded file is not a valid ZIP file. Please check the file and try again.")
                    return
                except Exception as e:
                    st.error(f"Error extracting ZIP file: {str(e)}")
                    return
                
                # Find all PDF files, ignoring macOS hidden files and directories
                pdf_files = []
                for root, dirs, files in os.walk(temp_dir):
                    # Skip macOS metadata directories
                    if "__MACOSX" in root:
                        continue
                    
                    for file in files:
                        # Skip hidden files and macOS metadata files
                        if file.startswith('.') or file.startswith('._'):
                            continue
                        if file.lower().endswith('.pdf'):
                            pdf_files.append(os.path.join(root, file))
                
                if not pdf_files:
                    st.warning("No PDF files found in the uploaded ZIP file.")
                    return
                
                st.write(f"Found {len(pdf_files)} PDF files.")
                
                # Process button
                if st.button("Process Resumes"):
                    # Check for API key
                    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-api-key-here":
                        st.error("Please update the OPENAI_API_KEY variable in the code with your actual API key.")
                        return
                    
                    # Initialize progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process each PDF
                    results = []
                    
                    for i, pdf_file in enumerate(pdf_files):
                        status_text.text(f"Processing {os.path.basename(pdf_file)}... ({i+1}/{len(pdf_files)})")
                        progress_bar.progress((i) / len(pdf_files))
                        
                        # Extract text from PDF
                        pdf_text = extract_text_from_pdf(pdf_file)
                        
                        if pdf_text:
                            # Analyze with LangChain using GPT-4o
                            analysis = analyze_resume(pdf_text, ranking_criteria)
                            
                            # Add filename
                            analysis['filename'] = os.path.basename(pdf_file)
                            results.append(analysis)
                            
                            # Add small delay to avoid rate limits
                            time.sleep(0.5)
                    
                    # Update progress
                    progress_bar.progress(1.0)
                    status_text.text("Processing complete!")
                    
                    # Sort results by match score
                    sorted_results = sorted(results, key=lambda x: x.get('match_score', 0), reverse=True)
                    
                    # Display results
                    st.subheader("Ranked Results")
                    
                    # Convert to DataFrame for better display
                    df_data = []
                    for r in sorted_results:
                        df_data.append({
                            'Rank': len(df_data) + 1,
                            'File Name': r.get('filename', 'Unknown'),
                            'Candidate Name': r.get('candidate_name', 'Unknown'),
                            'Match Score': r.get('match_score', 0),
                            'Relevant Skills': ', '.join(r.get('relevant_skills', [])) if isinstance(r.get('relevant_skills', []), list) else r.get('relevant_skills', 'N/A'),
                            'Education': r.get('education', 'N/A'),
                        })
                    
                    results_df = pd.DataFrame(df_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Detailed view for each resume
                    st.subheader("Detailed Analysis")
                    
                    for i, result in enumerate(sorted_results):
                        with st.expander(f"{i+1}. {result.get('filename', 'Unknown')} - Score: {result.get('match_score', 0)}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Candidate Information")
                                st.markdown(f"**Name:** {result.get('candidate_name', 'Unknown')}")
                                st.markdown(f"**Education:** {result.get('education', 'N/A')}")
                                
                                st.markdown("#### Relevant Skills")
                                if isinstance(result.get('relevant_skills', []), list):
                                    for skill in result.get('relevant_skills', []):
                                        st.markdown(f"- {skill}")
                                else:
                                    st.markdown(result.get('relevant_skills', 'N/A'))
                            
                            with col2:
                                st.markdown("#### Relevant Experience")
                                st.markdown(result.get('relevant_experience', 'N/A'))
                                
                                st.markdown("#### Reasoning")
                                st.markdown(result.get('reasoning', 'N/A'))
                    
                    # Option to download results as CSV
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results as CSV",
                        csv,
                        "resume_ranking_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
            except Exception as e:
                st.error(f"Error processing zip file: {str(e)}")
            
            finally:
                # Clean up
                shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
