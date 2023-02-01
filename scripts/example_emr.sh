#!/bin/bash

epoch=49
lr=0.02
cuda=0
expname="EX01_"

yaml='EMR_train_NG06_SS07_cfg.yaml'
seed=55
data_location="./dataset/EMR/"
data_dir='NG_resized'

# With SymmNet and scaling
CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'None' --method SymmNetsV1 --cfg ./experiments/configs/EMR/${yaml} \
--set DATASET.DATAROOT ${data_location}${data_dir} TRAIN.BASE_LR $lr MODEL.BP False SEED $seed TRAIN.MAX_EPOCH ${epoch} \
INV.ALPHA 10.0
