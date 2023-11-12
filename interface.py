import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModelForCausalLM
from function_qa.qg_t5_large import generate_question


tokenizer = LlamaTokenizer.from_pretrained("/home/lizekai/vicuna-7b-v1.3")
tokenizer.pad_token_id = 0

vicuna = LlamaForCausalLM.from_pretrained(
    "/home/lizekai/vicuna-7b-v1.3",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
vicuna_lora = PeftModelForCausalLM.from_pretrained(
    vicuna,
    "function_qa/vicuna-lora-book-qa",
    device_map="auto"
)

qg_tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer")
qg_model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer",
                                                 device_map="auto")


def question_answering(user_question, context, max_length, answer_length):
    prompt = f"""Below is a question paired with context. Please write a response to answer the question
    ###Question: {user_question}
    ###Context: {context}
    ###Response: 
    """
    input_ids = tokenizer(
        prompt,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).input_ids[0].cuda()

    output_ids = vicuna_lora.generate(
        input_ids=input_ids,
        max_new_tokens=answer_length
    )

    answer = tokenizer.decode(output_ids[0][len(input_ids):], skip_special_tokens=True)
    return answer


def question_generation(content, ques_num):
    content_words = content.split()
    chunk_size = int(len(content_words) / ques_num)
    qa_pairs = []
    for i in range(ques_num):
        chunk = content_words[i * chunk_size: (i + 1) * chunk_size]
        chunk_text = ' '.join(chunk)
        question, answer = generate_question(qg_tokenizer, qg_model, chunk_text)

        if question is not None:
            qa_pairs.append({'question': question, 'answer': answer})

    return qa_pairs


