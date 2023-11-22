from functions.qg_t5_large import generate_question
from prompts import prompt_templates


def summarization(tokenizer, model, text):

    inputs = tokenizer(
        prompt_templates['summarization'].format(text=text), 
        padding='max_length', 
        truncation=True, 
        return_tensors="pt"
    )
    input_ids = inputs.input_ids.cuda()
    outputs = model.generate(input_ids, max_new_tokens=100, min_new_tokens=30)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    

def question_answering(tokenizer, model, user_question, context):

    inputs = tokenizer(
        prompt_templates['question-answering'].format(context=context, question=user_question),
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = inputs.input_ids.cuda()
    outputs = model.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def question_generation(qg_tokenizer, qg_model, content, ques_num):

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


