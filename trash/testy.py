import streamlit as st
import pdfplumber
from transformers import pipeline

# Load a question-answering model
qa_pipeline = pipeline("question-answering")

st.title("Local Knowledge Base")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
question = st.text_input("Ask a question about the PDF content")

if uploaded_file and question:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    # Use the question-answering model
    answer = qa_pipeline(question=question, context=text)
    st.write(answer['answer'])
