TEST_DATA_PATH="val_data.json"
MODEL_PATH="/home/lizekai/llama-2-7b-hf"
LORA_PATH="/home/lizekai/LiteraLink/function_qa/llama2-lora-book-qa/checkpoint-1000"

nohup python3 inference.py \
--test_data_path $TEST_DATA_PATH \
--model_path $MODEL_PATH \
--lora_path $LORA_PATH \
--batch_size 4 \
--max_new_tokens 20 \
--num_beams 3 \
> logs/inference_20231110.log 2>&1 &