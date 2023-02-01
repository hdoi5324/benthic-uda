#!/bin/bash

data_location="./dataset/SHR/"
data_dir_1='crop_224_resize_224'
data_dir_2='AE2000F_RESIZED'
epoch=49
lr=0.02
warm_epoch=3
warm_lr=0.1
post_epoch=49
cuda=0
seed=55
expname="SH00_"
yamls=('SHR_train_AE2000_TUNASAND_cfg.yaml' 'SHR_train_TUNASAND_AE2000_cfg.yaml' )
resumes=("./experiments/output/SHR_AE2000_TUNASAND/${expname}SHR_UTOK AE2000F AUV2UTOK TunaSand AUV_SymmNetsV1_SourceOnlyclosed/ckpt_last.resume" \
"./experiments/output/SHR_TUNASAND_AE2000/${expname}SHR_UTOK TunaSand AUV2UTOK AE2000F AUV_SymmNetsV1_SourceOnlyclosed/ckpt_last.resume" )
resumes_none=("./experiments/output/SHR_AE2000_TUNASAND/${expname}SHR_UTOK AE2000F AUV2UTOK TunaSand AUV_SymmNetsV1_Noneclosed/ckpt_last.resume" \
"./experiments/output/SHR_TUNASAND_AE2000/${expname}SHR_UTOK TunaSand AUV2UTOK AE2000F AUV_SymmNetsV1_Noneclosed/ckpt_last.resume" )

for j in 0 1
do
  yaml="${yamls[$j]}"
  resume="${resumes[$j]}"
  resume_none="${resumes_none[$j]}"
  echo $lr $resume $resume_none $yaml
    for data_dir in 'crop_224_resize_224' 'AE2000F_RESIZED'
    do
      echo $data_dir
      # NO SCALING
      # Nothing
      CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'SourceOnly' --method SymmNetsV1 --cfg ./experiments/configs/SHR/${yaml} \
      --set DATASET.DATAROOT ${data_location}${data_dir} TRAIN.BASE_LR $lr MODEL.BP False SEED $seed TRAIN.MAX_EPOCH ${epoch} \
      INV.ALPHA 10.0
      #  - with Symmnet (NO BP)
      CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'None' --method SymmNetsV1 --cfg ./experiments/configs/SHR/${yaml} \
      --set DATASET.DATAROOT ${data_location}${data_dir} TRAIN.BASE_LR $lr MODEL.BP False SEED $seed TRAIN.MAX_EPOCH ${epoch} \
      INV.ALPHA 10.0

      #BP True
      # With BP & no SymmNet - warmup
      CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'SourceOnly' --method SymmNetsV1 --cfg ./experiments/configs/SHR/${yaml} \
      --set DATASET.DATAROOT ${data_location}${data_dir} MODEL.BP True SEED $seed TRAIN.MAX_EPOCH ${warm_epoch} \
      INV.ALPHA 10.0 BP.Q 4 BP.AB 64 BP.R 16 TRAIN.SAVING True TRAIN.LR_SCHEDULE 'fix' TRAIN.BASE_LR ${warm_lr}
      # With BP & no SymmNet - finish
      CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'SourceOnly' --method SymmNetsV1 --cfg ./experiments/configs/SHR/${yaml} \
      --set DATASET.DATAROOT ${data_location}${data_dir} TRAIN.BASE_LR $lr MODEL.BP True SEED $seed TRAIN.MAX_EPOCH ${post_epoch} \
      INV.ALPHA 10.0 BP.Q 4 BP.AB 64 BP.R 16 RESUME "${resume}"
      # With BP and SymmNet - warmup
      CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'None' --method SymmNetsV1 --cfg ./experiments/configs/SHR/${yaml} \
      --set DATASET.DATAROOT ${data_location}${data_dir} MODEL.BP True SEED $seed TRAIN.MAX_EPOCH ${warm_epoch} \
      INV.ALPHA 10.0 BP.Q 4 BP.AB 64 BP.R 16 TRAIN.SAVING True TRAIN.LR_SCHEDULE 'fix' TRAIN.BASE_LR ${warm_lr}
      # With BP and SymmNet - finish
      CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'None' --method SymmNetsV1 --cfg ./experiments/configs/SHR/${yaml} \
      --set DATASET.DATAROOT ${data_location}${data_dir} TRAIN.BASE_LR $lr MODEL.BP True SEED $seed TRAIN.MAX_EPOCH ${post_epoch} \
      INV.ALPHA 10.0 BP.Q 4 BP.AB 64 BP.R 16 RESUME "${resume_none}"
  done
done