# pip install streamlit
# pip install transformers datasets torch
# pip install rouge-score

import PyPDF2
import io
from docx import Document
from PyPDF2.errors import PdfReadError
import re
from transformers import pipeline
import streamlit as st

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or "" 
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces/newlines
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = "Your long document text goes here..."
summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
print(summary[0]['summary_text'])

def summarize_document(doc_text):
    # Clean text
    cleaned_text = clean_text(doc_text)
    # Summarize
    summary = summarizer(cleaned_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Title of the app
st.title("Smart Document Summarizer")

# Upload file
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    # Extract and clean text
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(uploaded_file)
    else:
        text = uploaded_file.getvalue().decode("utf-8")

    # Summarize the text
    summary = summarize_document(text)

    # Show the summary
    st.write("### Summary")
    st.write(summary)

#streamlit run your_script.py