import torch
import re
import string
import collections
import argparse
from tqdm import tqdm
from datasets import load_dataset
from construct_dataset import BookQADataset
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import (
    prepare_model_for_int8_training, PeftModelForCausalLM
)

tokenizer = LlamaTokenizer.from_pretrained("/home/lizekai/llama-2-7b-hf")
tokenizer.pad_token_id = 0


def load_model(args):
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=args.use_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if args.use_8bit is True:
        model = prepare_model_for_int8_training(model)
    if args.lora_path is not None:
        model = PeftModelForCausalLM.from_pretrained(model, args.lora_path, device_map="auto")
    return tokenizer, model


def generate_prompt(instruction, input):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = f"""Below is a question paired with an input that provides further context. Write a response that fills in the '_'.
    ### Instruction: {instruction}
    ### Input: {input}
    ### Response: """
    return user_prompt


def load_test_data(args, tokenizer):
    test_data = load_dataset("json", data_files=args.test_data_path)["train"]
    print(len(test_data))
    test_dataset = BookQADataset(tokenizer, test_data, args.max_length, args.section_length, args.max_sections, do_qg=False)
    print(len(test_dataset))
    return test_dataset

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_scores(labels, preds):
    exact_scores = 0.0
    f1_scores = 0.0
    for label, pred in zip(labels, preds):
        f1_score = compute_f1(label, pred)
        em = compute_exact(label, pred)

        f1_scores += f1_score
        exact_scores += em
    avg_f1_scores = f1_scores / len(labels)
    avg_em_scores = exact_scores / len(labels)
    return avg_em_scores, avg_f1_scores


def collate_fn(batch):
    input_ids = []
    labels = []
    for inst, _input, output in batch:
        prompt = generate_prompt(inst, _input)
        input_ids.append(
            tokenizer(prompt, max_length=512, truncation=True, padding='max_length', return_tensors='pt').input_ids[0]
        )
        labels.append(output)
    # print(input_ids)
    input_ids = torch.stack(input_ids)
    return input_ids, labels


@torch.inference_mode()
def test(args):
    tokenizer, model = load_model(args)
    test_dataset = load_test_data(args, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    print(len(test_loader))
    all_labels, all_preds = [], []
    for input_ids, labels in tqdm(test_loader):
        
        input_ids = input_ids.to()
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams
        )

        preds = tokenizer.batch_decode(output_ids[:, len(input_ids[0]):], skip_special_tokens=True)
        all_preds.extend(preds)
        all_labels.extend(labels)
    print("Number of predicitions: ", len(all_preds))
    print("Number of labels: ", len(all_labels))
    
    avg_em_scores, avg_f1_scores = get_scores(all_labels, all_preds)
    print("Exact Match: {:.3f}  F1 score: {:.3f}".format(avg_em_scores, avg_f1_scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default=False)
    parser.add_argument("--test_data_path", type=str, default="/path/to/data")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_8bit", type=bool, default=True)
    parser.add_argument("--max_sections", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--section_length", type=int, default=200)
    args = parser.parse_args()

    test(args)