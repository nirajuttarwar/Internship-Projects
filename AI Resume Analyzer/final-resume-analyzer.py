import streamlit as st
import PyPDF2
import docx2txt
import pytesseract
from PIL import Image
import io
import os
import tempfile
import logging
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Set page config
st.set_page_config(page_title='Resume Analyzer AI', layout="wide")

class ResumeParser:
    MAX_FILE_SIZE = 8 * 1024 * 1024  # 8 MB
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
            raise ValueError(f"File size exceeds the maximum limit of 8 MB. Your file size: {file.size / (1024 * 1024):.2f} MB")

        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            text = ResumeParser.extract_text_from_pdf(file)
        elif file_extension in ['doc', 'docx']:
            text = ResumeParser.extract_text_from_doc(file)
        elif file_extension in ['txt', 'text']:
            text = file.getvalue().decode('utf-8')
        elif file_extension in ['png', 'jpg', 'jpeg']:
            text = ResumeParser.extract_text_from_image(file)
        else:
            raise ValueError("Unsupported file format")
        
        if not ResumeParser.check_word_count(text):
            raise ValueError(f"Word count exceeds the maximum limit of 2 million words. Your word count: {len(text.split())}")
        
        return text

    @staticmethod
    def extract_text_from_pdf(file):
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
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
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    def generate_chunks(self, text, chunk_size=500):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def truncate_output(self, text):
        words = text.split()
        if len(words) > self.MAX_OUTPUT_WORDS:
            truncated_text = ' '.join(words[:self.MAX_OUTPUT_WORDS])
        else:
            truncated_text = text

        if len(truncated_text.encode('utf-8')) > self.MAX_OUTPUT_SIZE:
            truncated_text = truncated_text.encode('utf-8')[:self.MAX_OUTPUT_SIZE].decode('utf-8', 'ignore')

        return truncated_text

    def gemini_query(self, prompt):
        try:
            response = self.model.generate_content(prompt)
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
        chunks = self.generate_chunks(self.resume_text)
        prompt = f'''act as a expert in the field of the given resume and Analyze the following resume and divide it into standard resume sections such as:
                    1. Summary or Objective
                    2. Work Experience
                    3. Education
                    4. Skills
                    ....and more Additional Sections like (Projects, Certifications, Languages, etc...)
                    and give each sections strengths and weaknesses with recommendations for improvements
                    (don't give for all give only for needed sections which will impact resume).
                    and if there is no sections like resume give advice to the user to add resume in standard format.
                    
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {" ".join(chunks)}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return self.gemini_query(prompt)

    def get_job_titles(self):
        summary = self.get_summary()
        prompt = f'''What are the job roles I can apply to on LinkedIn based on the following resume analysis?
                    
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {summary}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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
            resume_chunks = self.generate_chunks(self.resume_text)
            job_chunks = self.generate_chunks(job_description)

            # Construct the prompt for analysis
            prompt = f'''act as a Application tracking system (ATS) and Analyze the given resume against the job description to determine how well the resume fits the job requirements.
                        The similarity score between the resume and job description is {similarity_score * 100:.2f}%.
                        (dont show to UI about semantic similarity only give your analysis of matching resume with job description)
                        Provide a detailed analysis in the following format:
                        1. Overall match percentage: [Interpret the similarity score and provide a percentage]
                        2. Key skills/keywords found in both the resume and job description: [List the matching keywords]
                        3. Important keywords/skills from the job description missing in the resume: [List missing keywords]
                        4. Recommendations for improvement: [Provide specific suggestions to better align the resume with the job description]

                        Resume:
                        """
                        {" ".join(resume_chunks)}
                        """

                        Job Description:
                        """
                        {" ".join(job_chunks)}
                        """
                        '''

            # Generate the analysis using Gemini
            analysis = self.gemini_query(prompt)
            return analysis

        except Exception as e:
            logger.error(f"Error in job alignment analysis: {str(e)}")
            return self.truncate_output(f"An error occurred during job alignment analysis: {str(e)}")

def main():
    st.markdown("<h1 style='text-align: center; color: #4A4A4A;'>AI-Powered Resume Analyzer</h1>", unsafe_allow_html=True)

    if 'resume_analyzer' not in st.session_state:
        st.session_state['resume_analyzer'] = None
    if 'analysis_result' not in st.session_state:
        st.session_state['analysis_result'] = None
    if 'job_description' not in st.session_state:
        st.session_state['job_description'] = ""

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<h3 style='color: #4A4A4A;'>Upload Your Resume</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(label='Analyze Your Resume With Gemini', type=['pdf', 'docx', 'doc', 'txt'])
        
        if uploaded_file is not None:
            try:
                with st.spinner('Processing your resume...'):
                    resume_text = ResumeParser.parse_resume(uploaded_file)
                    st.session_state['resume_analyzer'] = ResumeAnalyzer(resume_text)
                st.success("Resume uploaded and processed successfully!")
            except Exception as e:
                st.error(str(e))

    if st.session_state['resume_analyzer']:
        st.markdown("<h3 style='color: #4A4A4A;'>Analysis Options</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Get Resume Analysis"):
                with st.spinner('Analyzing your resume...'):
                    try:
                        summary = st.session_state['resume_analyzer'].get_summary()
                        st.session_state['analysis_result'] = ("Resume Analysis", summary)
                    except Exception as e:
                        st.error(f"An error occurred during resume analysis: {str(e)}")
                        logging.error(f"Resume analysis error: {str(e)}", exc_info=True)

        with col2:
            if st.button("Get Suggested Job Titles"):
                with st.spinner('Generating job title suggestions...'):
                    try:
                        job_titles = st.session_state['resume_analyzer'].get_job_titles()
                        st.session_state['analysis_result'] = ("Suggested Job Titles", job_titles)
                    except Exception as e:
                        st.error(f"An error occurred while suggesting job titles: {str(e)}")
                        logging.error(f"Job title suggestion error: {str(e)}", exc_info=True)

        with col3:
            st.session_state['job_description'] = st.text_area("Enter the job description", 
                                                               value=st.session_state['job_description'], 
                                                               height=100)
            if st.button("Analyze Job Alignment"):
                if st.session_state['job_description']:
                    with st.spinner('Analyzing job alignment...'):
                        try:
                            alignment_analysis = st.session_state['resume_analyzer'].get_job_alignment(st.session_state['job_description'])
                            st.session_state['analysis_result'] = ("Job Alignment Analysis", alignment_analysis)
                        except Exception as e:
                            st.error(f"An error occurred during job alignment analysis: {str(e)}")
                            logging.error(f"Job alignment analysis error: {str(e)}", exc_info=True)
                else:
                    st.warning("Please enter a job description to analyze alignment.")

        # Display the analysis result in a full-width section below the buttons
        if st.session_state['analysis_result']:
            st.markdown("<hr>", unsafe_allow_html=True)
            result_title, result_content = st.session_state['analysis_result']
            st.markdown(f"<h2 style='color: #4A4A4A;'>{result_title}</h2>", unsafe_allow_html=True)
            st.markdown(f"<div class='analysis-result'>{result_content}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
