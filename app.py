import torch
import streamlit as st
from textExtraction import *
from summary import *
from interface import *
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

# vicuna = LlamaForCausalLM.from_pretrained(
#     "/home/lizekai/vicuna-7b-v1.3",
#     load_in_8bit=True,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
# vicuna_lora = PeftModelForCausalLM.from_pretrained(
#     vicuna,
#     "function_qa/vicuna-lora-book-qa",
#     device_map="auto"
# )

st.set_page_config(page_title="LiteraLink")
st.header("LiteraLink: PDF Local Knowledge Base")
# upload the file
pdf = st.file_uploader("Upload your PDF file", type="pdf")

device = torch.device("cuda:0")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)

qg_tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer")
qg_model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer").to(device)

if pdf:
    text, num_pages = extract_text(pdf)
    page = st.number_input('Input the page that you are checking', min_value=0, max_value=num_pages, step=1)
    text, num_pages = extract_text(pdf, start_page=page, end_page=page)

    if st.button('Submit') and text and page:

        with st.spinner('Generating your summary...'):
            # summary = summarize_text(text)
            summary = summarization(tokenizer, model, text)
            st.success('Done! Here is your summary:')
        st.write(summary)

        if st.button('Ask a question'):
            user_question = st.text_input("Ask me anything about the content：")
            st.write(user_question)
            if user_question:
                with st.spinner('Getting the answer...'):
                    response = question_answering(tokenizer, model, user_question, text[:1000])
                    st.success('Done!')
                st.balloons()
                st.write(response)

        if st.button('Generate some questions'):
            ques_num = st.number_input('How many questions do you want to generate?', min_value=1, max_value=10, step=1)
            if ques_num:
                with st.spinner('Generating questions...'):
                    qa_pairs = question_generation(qg_tokenizer, qg_model, text, ques_num)
                    st.success('Done!')
                st.balloons()

                for item in qa_pairs:
                    with st.expander(item['question']):
                        st.write(item['answer'])


else:
    st.write("Please upload a PDF file")