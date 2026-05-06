#!/bin/bash

RUN_NAME="fairy"

DATASET_DIR="./data"
OUTPUT_DIR="./training_output"

BATCH_SIZE=8
EPOCHS=10
LEARNING_RATE=5e-5   
WEIGHT_DECAY=0.05
NUM_WORKERS=6


echo "Avvio training: $RUN_NAME..."
echo "Configurazione: BS=$BATCH_SIZE, LR=$LEARNING_RATE, Epochs=$EPOCHS"

python src/train.py \
    --name "$RUN_NAME" \
    --data_dir "$DATASET_DIR" \
    --save_dir "$OUTPUT_DIR" \
    --resume "./dice_last_ckpt/last_checkpoint.pth" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --wd $WEIGHT_DECAY \
    --num_workers $NUM_WORKERS \
    --grad_ckpoint true \
    --device "cuda" \

echo "Training completato."