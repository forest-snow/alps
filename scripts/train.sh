### change these variables if needed
DATA_DIR=data
TASK_NAME=imdb
MODEL_TYPE=bert
MODEL_NAME=bert-base-uncased
SEED=125
OUTPUT=models/$SEED/$TASK_NAME/base
### end

python -m src.train \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --data_dir $DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT \
    --seed $SEED \
    --base_model $MODEL_NAME

