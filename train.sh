#!/bin/bash

# 训练设置
EPOCHS=3
LR=0.01
BATCH_SIZE=32
IMG_SIZE=100
FRUITS360_CLASSES=131
FRC_CLASSES=2

# 备选模型
declare -a MODELS=("resnet18" "resnet34" "densenet121" "densenet161"
                   "efficientnet_b0" "efficientnet_b1" "efficientnet_b3"
                   "efficientnet_b5" "efficientnet_b7" "mobilenet_v2"
                   "mobilenet_v3_small" "mobilenet_v3_large")

# 循环每个模型，分别在两个数据集上训练
for model in "${MODELS[@]}"; do
    # 训练 fruits360 数据集
    echo "Training $model on fruits360"
    python train.py --input-size $BATCH_SIZE 3 $IMG_SIZE $IMG_SIZE -lr $LR --epoch $EPOCHS --dataset fruits360 --num-classes $FRUITS360_CLASSES --model $model

    # 训练 FRC 数据集
    echo "Training $model on FRC"
    python train.py --input-size $BATCH_SIZE 3 $IMG_SIZE $IMG_SIZE -lr $LR --epoch $EPOCHS --dataset FRC --num-classes $FRC_CLASSES --model $model
done
