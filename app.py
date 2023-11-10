import streamlit as st
from response import *
from textExtraction import *

st.set_page_config(page_title="LiteraLink")
st.header("LiteraLink: PDF Local Knowledge Base")
# upload the file
pdf = st.file_uploader("Upload your PDF file", type="pdf")

if pdf is not None:
    done = extract_text(pdf)

    if done:
        user_question = st.text_input("Ask me anything about the content：")
        st.write(user_question)

        if user_question is not None:

            response = generate_response("sample_pdf/output.txt", user_question)

            if response is not None:

                st.balloons()

                st.write(response)
else:
    st.write("Please upload a PDF file")