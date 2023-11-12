import streamlit as st
from responseGeneration import *
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

        user_question = st.text_input("Ask me anything about the content：")

        st.write(user_question)

        if user_question:
            with st.spinner('Getting the answer...'):
                response = generate_response(text, user_question)
                st.success('Done!')
            st.balloons()
            st.write(response)
else:
    st.write("Please upload a PDF file")