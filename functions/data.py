from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from nltk.tokenize import word_tokenize

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")


def load_data(dataset_name):
    # train_ds = load_dataset("csv", data_files=train_path)
    # val_ds = load_dataset("csv", data_files=val_path)
    train_ds, val_ds = load_dataset(dataset_name, split=['train', 'validation'])
    return train_ds, val_ds


def process_data(example):

    text = example['chapter'][:5000]
    summary = example['summary_text']

    processed_text = "Write a summary for this text: {text}".format(text=text)
    tokenized_text = tokenizer(
        processed_text,
        max_length=1024,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    text_input_ids, text_attention_mask = tokenized_text['input_ids'], tokenized_text['attention_mask']

    tokenized_summary = tokenizer(
        summary,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    label_input_ids, label_attention_mask = tokenized_summary['input_ids'], tokenized_summary['attention_mask']
    label_with_ignore_index = []
    for label in label_input_ids:
        label = [l if l != 0 else -100 for l in label]
        label_with_ignore_index.append(label)

    return {
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'label_input_ids': label_with_ignore_index,
        'label_attention_mask': label_attention_mask
    }


def process_data_for_trainer(example):

    text = example['chapter'][:5000]
    summary = example['summary_text']
    model_input = {}
    processed_text = "Write a summary for this text: {text}".format(text=text)
    inputs = tokenizer(
        processed_text,
        max_length=1024,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    model_input['input_ids'] = inputs['input_ids'][0]
    model_input['attention_mask'] = inputs['attention_mask'][0]

    tokenized_summary = tokenizer(
        summary,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    model_input['labels'] = tokenized_summary['input_ids'][0]
    return model_input


def create_data_loader(config, train_ds, val_ds):
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def data_helper(config):
    train_ds, val_ds = load_data(config.dataset)
    train_ds = train_ds.map(process_data, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(process_data, batched=True, remove_columns=val_ds.column_names)

    train_ds.set_format(type='torch')
    val_ds.set_format(type='torch')

    train_loader, val_loader = create_data_loader(config, train_ds, val_ds)
    return train_loader, val_loader

def data_helper_for_trainer(config):
    train_ds, val_ds = load_data(config.dataset)
    train_ds = train_ds.map(process_data_for_trainer)
    val_ds = val_ds.map(process_data_for_trainer)

    # train_ds.set_format(type='torch')
    # val_ds.set_format(type='torch')

    return train_ds, val_ds


def generate_sample(config):
    train_loader, val_loader = data_helper(config)
    batch = next(iter(train_loader))
    decoded_text = tokenizer.decode(batch['text_input_ids'][0])
    decoded_summary = tokenizer.decode(batch['label_input_ids'][0])
    print(decoded_text)
    print(decoded_summary)


def dataset_statistics(dataset):
    max_text_length = 0
    max_summary_length = 0
    total_text_length = 0
    total_summary_length = 0
    num_of_examples = 0

    for example in tqdm(dataset):
        text_len = len(word_tokenize(example['text']))
        summary_len = len(word_tokenize(example['summary']))
        max_text_length = max(max_text_length, text_len)
        max_summary_length = max(max_summary_length, summary_len)
        total_text_length += text_len
        total_summary_length += summary_len
        num_of_examples += 1

    avg_text_length = total_text_length / num_of_examples
    avg_summary_length = total_summary_length / num_of_examples

    return max_text_length, max_summary_length, avg_text_length, avg_summary_length


if __name__ == '__main__':
    pass
    # generate_sample()
