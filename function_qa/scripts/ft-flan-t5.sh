DATASET_NAME="kmfoda/booksum"
SAVE_PATH="flan-t5-book"
MODEL_PATH="google/flan-t5-large"

nohup python3 finetune-flan-t5.py \
--save_path $SAVE_PATH \
--model_path $MODEL_PATH \
--dataset $DATASET_NAME \
--num_epochs 10 \
--max_length 1024 \
--batch_size 16 \
--init_lr 5e-5 \
> logs/finetune_flan_t5_20231114.log 2>&1 &