import torch
import streamlit as st
from textExtraction import *
from summary import *
from interface import *
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="LiteraLink")
st.header("LiteraLink: PDF Local Knowledge Base")
# upload the file
pdf = st.file_uploader("Upload your PDF file", type="pdf")

device = torch.device("cuda")
tokenizer = T5Tokenizer.from_pretrained("logits/flan-t5-booksum")
model = T5ForConditionalGeneration.from_pretrained("logits/flan-t5-booksum").to(device)

qg_tokenizer = AutoTokenizer.from_pretrained("logits/flan-t5-booksum-qg")
qg_model = AutoModelForSeq2SeqLM.from_pretrained("logits/flan-t5-booksum-qg").to(device)

if pdf:
    text, num_pages = extract_text(pdf)
    page = st.number_input('Input the page that you are checking', min_value=0, max_value=num_pages, step=1)
    text, num_pages = extract_text(pdf, start_page=page, end_page=page)

    if st.button('Submit') and text and page:
        submitted = True
        with st.spinner('Generating your summary...'):
            summary = summarization(tokenizer, model, text)
            st.success('Done! Here is your summary:')
        st.write(summary)

    choice = st.radio('Choose an option:', ['Ask a question', 'Generate some questions'])

    if choice == 'Ask a question':
        user_question = st.text_input("Ask me anything about the content：")
        st.write(user_question)
        if user_question:
            with st.spinner('Getting the answer...'):
                response = question_answering(tokenizer, model, user_question, text[:1000])
                st.success('Done!')
            st.balloons()
            st.write(response)

    else:
        ques_num = st.number_input('How many questions do you want to generate?', min_value=1, max_value=10, step=1)
        if st.button('Confirm') and ques_num:
            with st.spinner('Generating questions...'):
                qa_pairs = question_generation(qg_tokenizer, qg_model, text, ques_num)
                st.success('Done!')
            st.balloons()

            for item in qa_pairs:
                with st.expander(item['question']):
                    st.write(item['answer'])


else:
    st.write("Please upload a PDF file")