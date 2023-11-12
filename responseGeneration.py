from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
from peft import PeftModelForCausalLM
import bitsandbytes
import accelerate

def load_model():
    # tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    # model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    base_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token='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', load_in_4bit=True)
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token='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')
    model = PeftModelForCausalLM.from_pretrained(base_model, "logits/llama2-7b-book-qa", load_in_4bit=True)
    return tokenizer, model

def generate_text(prompt, tokenizer, model, max_new_tokens=500):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_response(text, question):
    tokenizer, model = load_model()
    prompt = "Please answer the question according the context: \nQuestion: " + question + "\nContext: " + text
    generated_text = generate_text(prompt, tokenizer, model).split("\n")
    # print(generated_text)
    return '\n'.join(generated_text[3:])

if __name__ == "__main__":
    text = "This was early in March. During the next three months there was much secret activity. Major’s speech had given to the more intelligent animals on the farm a completely new outlook on life. They did not know when the Rebellion predicted by Major would take place, they had no reason for thinking that it would be within their own lifetime, but they saw clearly that it was their duty to prepare for it. The work of teaching and organising the others fell naturally upon the pigs, who were generally recognised as being the cleverest of the animals. Pre-eminent among the pigs were two young boars named Snowball and Napoleon, whom Mr. Jones was breeding up for sale. Napoleon was a large, rather fierce-looking Berkshire boar, the only Berkshire on the farm, not much of a talker, but with a reputation for getting his own way. Snowball was a more vivacious pig than Napoleon, quicker in speech and more inventive, but was not considered to have the same depth of character. All the other male pigs on the farm were porkers. The best known among them was a small fat pig named Squealer, with very round cheeks, twinkling eyes, nimble movements, and a shrill voice. He was a brilliant talker, and when he was arguing some difficult point he had a way of skipping from side to side and whisking his tail which was somehow very persuasive. The others said of Squealer that he could turn black into white."
    question = "What was the name of the boars?"
    print(generate_response(text, question))