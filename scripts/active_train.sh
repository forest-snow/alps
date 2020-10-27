set -e

### change these variables if needed
DATA_DIR=data
TASK_NAME=imdb
MODEL_TYPE=bert
MODEL_NAME=bert-base-uncased
SEED=125
COLDSTART=none
SAMPLING=rand
INCREMENT=100
MAX_SIZE=1000
### end

METHOD=${COLDSTART}-${SAMPLING}
MODEL_DIR=models/${SEED}/${TASK_NAME}
if [ "$COLDSTART" == "none" ]
then
    MODEL0=$MODEL_NAME
    START=0
    METHOD=${SAMPLING}
else
    MODEL0=${MODEL_DIR}/${COLDSTART}_${INCREMENT}
    START=$INCREMENT
fi

active (){
# 1=number of samples
# 2=model path
# 3=sampling method
echo -e "\n\nACQUIRING $1 SAMPLES\n\n"
python -m src.active \
    --model_type $MODEL_TYPE \
    --model_name_or_path $2 \
    --task_name $TASK_NAME \
    --data_dir $DATA_DIR/$TASK_NAME \
    --output_dir ${MODEL_DIR}/${3}_${1} \
    --seed $SEED \
    --query_size $INCREMENT \
    --sampling $SAMPLING \
    --base_model $MODEL_NAME \
    --per_gpu_eval_batch_size 32 \
    --max_seq_length 128
}

train (){
# 1 = number of samples
# 2 = output directory
echo -e "\n\nTRAINING WITH $1 SAMPLES\n\n"
python -m src.train \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir $DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $2 \
    --seed $SEED \
    --base_model $MODEL_NAME \
    --per_gpu_eval_batch_size 32 \
    --per_gpu_train_batch_size 32
}

f=$MODEL0
p=$(( $START + $INCREMENT ))
while [ $p -le $MAX_SIZE ]
do
    active $p $f $METHOD
    f=${MODEL_DIR}/${METHOD}_$p
    train $p $f
    p=$(( $p + $INCREMENT ))
done
