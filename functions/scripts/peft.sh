TRAIN_DATA_PATH="train_data.json"
VAL_DATA_PATH="val_data.json"
OUTPUT_PATH="llama2-lora-book-qa"
MODEL_PATH="/home/lizekai/llama-2-7b-hf"

nohup python3 peft_llama2.py \
--train_data_path $TRAIN_DATA_PATH \
--val_data_path $VAL_DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 1000 \
--micro_batch_size 16 \
> logs/peft_llama2_20231108.log 2>&1 &