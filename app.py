import streamlit as st

st.set_page_config(page_title="LiteraLink")
st.header("LiteraLink: PDF Local Knowledge Base")
# upload the file
pdf = st.file_uploader("Upload your PDF file", type="pdf")

user_question = st.text_input("Ask me anything about the content：")


st.balloons()

response = "response placeholder"

st.write(response)