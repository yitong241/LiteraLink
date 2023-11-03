import random
from datasets import load_dataset
from tqdm import tqdm
from qg_openai import generate_questions
from torch.utils.data import Dataset


class BookQADataset(Dataset):
    def __init__(self, tokenizer, data, section_length, max_sections, max_length):
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
                qa_pair = generate_questions(section_text,
                                             num_qns=1,
                                             difficulty_level=self.difficulty_levels[i % 3])
                if qa_pair is None:
                    continue
                question, answer = qa_pair[0]['question'], qa_pair[0]['answer']
                self.processed_data.append({
                    'context': section_text,
                    'question': question,
                    'answer': answer
                })
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
