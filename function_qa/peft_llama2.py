import os
import sys
import torch
import transformers
import argparse
from datasets import load_dataset
from construct_dataset import BookQADataset
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model
)


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(
                checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control


def generate_and_tokenize_prompt(data_point, tokenizer, max_length):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request.
    ### Instruction: {data_point["instruction"]}
    ### Input: {data_point["input"]}
    ### Response: """
    len_user_prompt_tokens = (
            len(
                tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
            )
            - 1
    )  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }


def compute_metrics(pred):
    pass


def prepare_model_and_tokenizer(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = 0

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=args.use_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if args.use_8bit is True:
        model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    model.print_trainable_parameters()
    return tokenizer, model


def prepare_data(args, tokenizer):
    train_data, val_data, _ = load_dataset("kmfoda/booksum", split=["train", "validation", "test"])
    train_data = train_data.select(range(args.train_data_size))
    val_data = val_data.select(range(args.val_data_size))

    train_dataset = BookQADataset(tokenizer, train_data, args.section_length, args.max_section, args.max_length)
    val_dataset = BookQADataset(tokenizer, val_data, args.section_length, args.max_section, args.max_length)

    train_dataset.map(generate_and_tokenize_prompt)
    val_dataset.map(generate_and_tokenize_prompt)

    return train_dataset, val_dataset


def train(args):
    tokenizer, model = prepare_model_and_tokenizer(args)
    train_dataset, val_dataset = prepare_data(args, tokenizer)

    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=args.epochs,
        max_steps=args.max_step,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=50,
        evaluation_strategy="steps" if args.test_size > 0 else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if args.test_size > 0 else None,
        save_steps=args.save_steps,
        output_dir=args.output_path,
        save_total_limit=30,
        load_best_model_at_end=True if args.test_size > 0 else False,
        ddp_find_unused_parameters=False if args.ddp else None,
        report_to="wandb" if args.wandb else [],
        ignore_data_skip=args.ignore_data_skip,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[SavePeftModelCallback]
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default=True)
    parser.add_argument("--data_path", type=str, default="/path/to/data")
    parser.add_argument("--output_path", type=str, default="/path/to/output")
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--train_data_size", type=int, default=1000)
    parser.add_argument("--val_data_size", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
    parser.add_argument("--ignore_data_skip", type=str, default="False")
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=int, default=0.5)
    parser.add_argument("--use_8bit", type=bool, default=True)
    parser.add_argument("--section_length", type=int, default=200)
    parser.add_argument("--max_sections", type=int, default=15)
    args = parser.parse_args()

    train(args)