import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import docx2txt
import pytesseract
import os
from dotenv import load_dotenv
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config at the very beginning
st.set_page_config(page_title='Resume Analyzer AI', layout="wide")

# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

# Check if API key is available
if not api_key:
    logger.error("Gemini API key not found in environment variables.")
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
else:
    genai.configure(api_key=api_key)

# Custom CSS
def load_css():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

class ResumeParser:
    MAX_FILE_SIZE = 30 * 1024 * 1024
    MAX_WORD_COUNT = 1500000
    
    @staticmethod
    def check_file_size(file):
        return file.size <= ResumeParser.MAX_FILE_SIZE
    
    @staticmethod
    def check_word_count(text):
        return len(text.split()) <= ResumeParser.MAX_WORD_COUNT

    @staticmethod
    def parse_resume(file):
        if not ResumeParser.check_file_size(file):
            raise ValueError(f"File size exceeds the maximum limit of 30 MB. Your file size: {file.size / (1024 * 1024):.2f} MB")

        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            return ResumeParser.extract_text_from_pdf(file)
        elif file_extension in ['doc', 'docx']:
            return ResumeParser.extract_text_from_doc(file)
        elif file_extension in ['txt', 'text']:
            return file.getvalue().decode('utf-8')
        elif file_extension in ['png', 'jpg', 'jpeg']:
            text = ResumeParser.extract_text_from_image(file)
        else:
            raise ValueError("Unsupported file format")
        

    @staticmethod
    def extract_text_from_pdf(file):
        text = ""
        try:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            
            # Extract text from images in the PDF
            for page in pdf_reader.pages:
                if '/XObject' in page['/Resources']:
                    xObject = page['/Resources']['/XObject'].get_object()
                    for obj in xObject:
                        if xObject[obj]['/Subtype'] == '/Image':
                            size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                            data = xObject[obj].get_data()
                            image = Image.frombytes('RGB', size, data)
                            image_text = ResumeParser.extract_text_from_image(io.BytesIO(data))
                            if image_text.strip():  # Only add non-empty text
                                text += "\n" + image_text
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {str(e)}")
        return text

    @staticmethod
    def extract_text_from_doc(file):
        text = docx2txt.process(file)
        # Extract text from images in the document
        temp_dir = tempfile.mkdtemp()
        try:
            temp_file = os.path.join(temp_dir, "temp_doc")
            with open(temp_file, "wb") as f:
                f.write(file.getvalue())
            doc = docx2txt.process(temp_file, temp_dir)
            for image_file in os.listdir(temp_dir):
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(temp_dir, image_file)
                    image_text = ResumeParser.extract_text_from_image(image_path)
                    if image_text.strip():  # Only add non-empty text
                        text += "\n" + image_text
        finally:
            # Clean up temporary files
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
        return text
    
    @staticmethod
    def extract_text_from_image(file):
        try:
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            image = Image.open(file)

            text = pytesseract.image_to_string(image)
            
            return text
        except Exception as e:
            logging.error(f"Error extracting text from image: {str(e)}")
            return ""

class ResumeAnalyzer:
    MAX_OUTPUT_WORDS = 13653
    MAX_OUTPUT_SIZE = 32.8 * 1024  # 32.8 KB
    
    def __init__(self, resume_text):
        self.resume_text = resume_text

    def generate_chunks(self, chunk_size=500):
        return [self.resume_text[i:i + chunk_size] for i in range(0, len(self.resume_text), chunk_size)]
    
    def truncate_output(self, text):
        words = text.split()
        if len(words) > self.MAX_OUTPUT_WORDS:
            truncated_text = ' '.join(words[:self.MAX_OUTPUT_WORDS])
        else:
            truncated_text = text

        if len(truncated_text.encode('utf-8')) > self.MAX_OUTPUT_SIZE:
            truncated_text = truncated_text.encode('utf-8')[:self.MAX_OUTPUT_SIZE].decode('utf-8', 'ignore')

        return truncated_text

    @staticmethod
    def gemini_query(self, prompt):
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return self.truncate_output(response.text)
        except Exception as e:
            logger.error(f"Error in Gemini query: {str(e)}")
            raise ValueError(f"An error occurred while processing your request: {str(e)}")
        
    def get_embedding(self, text):
        try:
            embedding = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="similarity"
            )
            return np.array(embedding['embedding'])
        except Exception as e:
            logger.error(f"Error in getting embedding: {str(e)}")
            raise ValueError(f"An error occurred while getting embedding: {str(e)}")
        
    def calculate_similarity(self, resume_embedding, job_description_embedding):
        resume_embedding = np.array(resume_embedding).reshape(1, -1)
        job_description_embedding = np.array(job_description_embedding).reshape(1, -1)
        similarity = cosine_similarity(resume_embedding, job_description_embedding)[0][0]
        return similarity

    def get_summary(self):
        chunks = self.generate_chunks()
        prompt = f'''Act as a Human Resource Manager having all technical and non-technical knowledge in the fields of Engineering 
                    like artificial intelligence, data science, computer science, information technology, cyber security, civil engineering, mechanical engineering, etc.
                    Analyze the text extracted from resume. Which are given in chunks.
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {" ".join(chunks)}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    You have done analysis as Human Resource Manager, so give the structured data format from extracted text in standard sections of resume.
                    Take below key words for reference for creating structured data format:
                    
                    •	**Personal Information**
                    •	Professional Summary
                    •	Educational Information
                    •	Technical Skills
                    •	Experience
                    •	Assessments /Certifications
                    •	Projects
                    •	Publications / Research
                    •	Co-Curricular Activities/ Volunteer Experience, etc.
                    '''
        return self.gemini_query(prompt)
    
    def get_SWS(self):
        summary = self.get_summary()
        prompt = f'''Act as a Human Resource Manager having all technical and non-technical knowledge in the fields of Engineering 
                    like artificial intelligence, data science, computer science, information technology, cyber security, civil engineering, mechanical engineering, etc.
                    Analyze the text extracted from resume. Which are given in chunks.
                    
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {summary}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""  
                                      
                    Identify and give the technical strengths that are boosting resume and make a standard list of them.
                    Now Identify and give weaknesses with recommendations for improvements (don't give for all give only for needed sections which will impact resume). 
                    and if there is no sections like resume give advice to the user to add resume in standard format.

                    '''
        return self.gemini_query(prompt)
    
    def get_job_alignment(self, job_description):
        try:
            # Get embeddings for resume and job description
            resume_embedding = self.get_embedding(self.resume_text)
            job_description_embedding = self.get_embedding(job_description)
            
            # Calculate similarity score
            similarity_score = self.calculate_similarity(resume_embedding, job_description_embedding)

            # Prepare chunks for analysis
            chunks = self.generate_chunks(self.resume_text)
            job_chunks = self.generate_chunks(job_description)

            prompt = f'''You are an experienced Human Resources Manager with extensive technical and non-technical knowledge across multiple engineering disciplines.
                        Act as expert Application Tracking System(ATS) and analyze the given summary of the resume against the job description to determine how well the resume fits the job requirements. 
                        The similarity score between the resume and job description is {similarity_score * 100:.2f}%. 
                        (dont show to UI about semantic similarity only give your analysis of matching resume with job description). 
                        Provide a overall detailed analysis. Take below points for reference:
                        1. Overall match percentage: [Interpret the similarity score and provide a percentage]
                        2. Key skills/keywords found in both the resume and job description: [List the matching keywords]
                        3. Important keywords/skills from the job description missing in the resume: [List missing keywords]
                        4. Recommendations for improvement: [Provide specific suggestions to better align the resume with the job description 

                        Resume:
                        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        {" ".join(chunks)}
                        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                        Job Description:
                        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        {job_description}
                        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        '''
            return self.gemini_query(prompt)
    
        except Exception as e:
                logger.error(f"Error in job alignment analysis: {str(e)}")
                return self.truncate_output(f"An error occurred during job alignment analysis: {str(e)}")
                    
        
    def get_job_titles(self):
        summary = self.get_summary()
        prompt = f'''What are the job roles I can apply to on LinkedIn based on the following resume summary?
                    
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {summary}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return self.gemini_query(prompt)

    

def main():
    load_css()
    st.markdown("<h1 class='main-title'>AI-Powered Resume Analyzer</h1>", unsafe_allow_html=True)

    if 'resume_uploaded' not in st.session_state:
        st.session_state['resume_uploaded'] = False
    if 'resume_analyzer' not in st.session_state:
        st.session_state['resume_analyzer'] = None

    if not st.session_state['resume_uploaded']:
        upload_resume()
    else:
        show_analysis_options()

def upload_resume():
    st.markdown("<h2 class='section-title'>Upload Your Resume</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(label='Analyze Your Resume', type=['pdf', 'docx', 'doc', 'txt'])
    if uploaded_file is not None:
        try:
            with st.spinner('Processing your resume...'):
                resume_text = ResumeParser.parse_resume(uploaded_file)
                st.session_state['resume_analyzer'] = ResumeAnalyzer(resume_text)
                st.session_state['resume_uploaded'] = True
            st.success("Resume uploaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(str(e))

def show_analysis_options():
    st.sidebar.title("Analysis Options")
    
    st.markdown("<div class='content-box'>", unsafe_allow_html=True)
    if st.sidebar.button("Summary", key="summary_btn"):
        show_summary()
        
    if st.sidebar.button("Strength and Weakness", key="sws_btn"):
        Show_SWS()
        
    if st.sidebar.button("Job Titles", key="job_titles_btn"):
        show_job_titles()
        
    if st.sidebar.button("Job Alignment", key="job_alignment_btn"):
        show_job_alignment()
        
    if st.sidebar.button("Upload New Resume", key="new_resume"):
        st.session_state['resume_uploaded'] = False
        st.session_state['resume_analyzer'] = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def show_summary():
    st.markdown("<h2 class='section-title'>Resume Summary</h2>", unsafe_allow_html=True)
    try:
        with st.spinner('Analyzing your resume...'):
            summary = st.session_state['resume_analyzer'].get_summary()
        st.markdown(f"<div class='summary-text'>{summary}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
def Show_SWS():
    st.markdown("<h2 class='section-title'>Resume Streangth and Weakness</h2>", unsafe_allow_html=True)
    try:
        with st.spinner('Analyzing your resume...'):
            Streangth_and_Weakness = st.session_state['resume_analyzer'].get_SWS()
        st.markdown(f"<div class='Streangth-and-Weakness'>{Streangth_and_Weakness}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def show_job_titles():
    st.markdown("<h2 class='section-title'>Suggested Job Titles</h2>", unsafe_allow_html=True)
    try:
        with st.spinner('Generating job title suggestions...'):
            job_titles = st.session_state['resume_analyzer'].get_job_titles()
        st.markdown(f"<div class='job-titles'>{job_titles}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def show_job_alignment():
    st.markdown("<h2 class='section-title'>Job Alignment Analysis</h2>", unsafe_allow_html=True)
    job_description = st.text_area("Enter the job description", height=200)
    if job_description:
        try:
            with st.spinner('Analyzing job alignment...'):
                alignment_analysis = st.session_state['resume_analyzer'].get_job_alignment(job_description)
            st.markdown(f"<div class='alignment-analysis'>{alignment_analysis}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a job description to analyze alignment.")

if __name__ == "__main__":
    main()
