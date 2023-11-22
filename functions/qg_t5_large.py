
def generate_question(tokenizer, model, context):

    inputs = tokenizer(context, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    question_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
    if len(question_answer.split(tokenizer.sep_token)) == 2:
        question, answer = question_answer.split(tokenizer.sep_token)
    else:
        question, answer = None, None

    return question, answer
