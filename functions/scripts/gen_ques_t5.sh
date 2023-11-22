MODEL_PATH="/home/lizekai/llama-2-7b-hf"
TRAIN_SAVE_PATH="train_data.json"
VAL_SAVE_PATH="val_data.json"

nohup python3 construct_dataset.py \
--model_path $MODEL_PATH \
--train_save_path $TRAIN_SAVE_PATH \
--val_save_path $VAL_SAVE_PATH \
--train_data_size 500 \
--val_data_size 100 \
--max_length 512 \
--max_sections 20 \
--section_length 200 \
> logs/question_generation_20231107.log 2>&1 &
