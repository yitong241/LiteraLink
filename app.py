import streamlit as st

st.set_page_config(page_title="LiteraLink")
st.header("LiteraLink: PDF Local Knowledge Base")
# upload the file
pdf = st.file_uploader("Upload your PDF file", type="pdf")
