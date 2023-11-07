import json
import time
import random
import argparse
from datasets import load_dataset
from tqdm import tqdm
# from qg_openai import generate_questions
from qg_t5_large import generate_question
from torch.utils.data import Dataset
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer")
model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer")

class BookQADataset(Dataset):
    def __init__(self, tokenizer, data, section_length, max_sections, max_length, mode):
        self.tokenizer = tokenizer
        self.data = data
        self.section_length = section_length
        self.max_sections = max_sections
        self.max_length = max_length
        self.difficulty_levels = ["easy", "medium", "hard"]
        self.processed_data = []
        self.inst_format_data = []
        self.input_data = []

        self.process_sections()

        if mode == "t5-large":
            self.qg_tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer")
            self.qg_model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer")

        self.process_format()

    def process_sections(self):
        print("Processing dataset...")
        for example in tqdm(self.data):
            book_context = example["chapter"]
            context_tokens = book_context.split()

            for i in range(self.max_sections):
                start_index = i * self.section_length
                end_index = (i + 1) * self.section_length
                if start_index >= len(context_tokens) or end_index >= len(context_tokens):
                    break
                section = context_tokens[start_index: end_index]
                section_text = ' '.join(section)
                # qa_pair = generate_questions(section_text,
                #                              num_qns=1,
                #                              difficulty_level=self.difficulty_levels[i % 3])
                # if qa_pair is None:
                #     continue
                # question, answer = qa_pair[0]['question'], qa_pair[0]['answer']
                question, answer = generate_question(self.qg_tokenizer, self.qg_model, section_text)
                self.processed_data.append({
                    'context': section_text,
                    'question': question,
                    'answer': answer
                })

            # time.sleep(10)
        random.shuffle(self.processed_data)

    def process_format(self):
        for example in self.processed_data:
            context = example['context']
            question = example["question"]
            answer = example["answer"]

            inst_format_example = {
                "instruction": question,
                "input": context,
                "output": answer,
            }
            self.inst_format_data.append(inst_format_example)

    def map(self, func):
        for data in self.inst_format_data:
            tokenized_data = func(data, self.tokenizer, self.max_length)
            self.input_data.append(tokenized_data)

    def __getitem__(self, item):
        if not self.input_data:
            return self.processed_data[item]
        return self.input_data[item]

    def __len__(self):
        return len(self.processed_data)

    def save_data(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.inst_format_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/path/to/data")
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    parser.add_argument("--train_save_path", type=str, default="/path/to/save")
    parser.add_argument("--val_save_path", type=str, default="/path/to/save")
    parser.add_argument("--train_data_size", type=int, default=1000)
    parser.add_argument("--val_data_size", type=int, default=100)
    parser.add_argument("--section_length", type=int, default=200)
    parser.add_argument("--max_sections", type=int, default=15)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = 0

    train_data, val_data, _ = load_dataset("kmfoda/booksum", split=["train", "validation", "test"])
    train_data = train_data.select(range(args.train_data_size))
    val_data = val_data.select(range(args.val_data_size))

    train_dataset = BookQADataset(tokenizer, train_data, args.section_length, args.max_sections, args.max_length)
    val_dataset = BookQADataset(tokenizer, val_data, args.section_length, args.max_sections, args.max_length)

    train_dataset.save_data(args.train_save_path)
    val_dataset.save_data(args.val_save_path)


