MODEL_PATH="/Users/zekaili/Documents/Pretrain-LM/vicuna-7b-v1.3"
TRAIN_SAVE_PATH="train_data_30.json"
VAL_SAVE_PATH="val_data_10.json"

nohup python3 construct_dataset.py \
--model_path $MODEL_PATH \
--train_save_path $TRAIN_SAVE_PATH \
--val_save_path $VAL_SAVE_PATH \
--train_data_size 30 \
--val_data_size 10 \
--max_length 512 \
--max_sections 10 \
--section_length 200 \
> logs/question_generation_20231107.log 2>&1 &
