from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
from peft import PeftModelForCausalLM

def load_model():
    # tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    # model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    base_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = PeftModelForCausalLM.from_pretrained(base_model, "logits/llama2-7b-book-qa")
    return tokenizer, model

def generate_text(prompt, tokenizer, model, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_response(text, question):
    tokenizer, model = load_model()
    prompt = "Please summarize the following text: " + text + "\nQuestion: " + question + "\nAnswer:"
    generated_text = generate_text(prompt, tokenizer, model)
    print("Generated Text:", generated_text)



if __name__ == "__main__":
    generate_response()