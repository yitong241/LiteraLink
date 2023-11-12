import streamlit as st
from textExtraction import *
from summary import *
from interface import *


st.set_page_config(page_title="LiteraLink")
st.header("LiteraLink: PDF Local Knowledge Base")
# upload the file
pdf = st.file_uploader("Upload your PDF file", type="pdf")


if pdf:
    text, num_pages = extract_text(pdf)
    page = st.number_input('Input the page that you are checking', min_value=0, max_value=num_pages, step=1)
    text, num_pages = extract_text(pdf, start_page=page, end_page=page)

    if st.button('Submit') and text and page:

        with st.spinner('Generating your summary...'):
            summary = summarize_text(text)
            st.success('Done! Here is your summary:')
        st.write(summary)

        if st.button('Ask a question'):
            user_question = st.text_input("Ask me anything about the content：")
            st.write(user_question)
            if user_question:
                with st.spinner('Getting the answer...'):
                    response = question_answering(user_question, text)
                    st.success('Done!')
                st.balloons()
                st.write(response)

        if st.button('Generate some questions'):
            ques_num = st.number_input('How many questions do you want to generate?', min_value=1, max_value=10, step=1)
            if ques_num:
                with st.spinner('Generating questions...'):
                    qa_pairs = question_generation(text, ques_num)
                    st.success('Done!')
                st.balloons()

                for item in qa_pairs:
                    with st.expander(item['question']):
                        st.write(item['answer'])


else:
    st.write("Please upload a PDF file")