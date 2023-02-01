#!/bin/bash

data_location="./dataset/EMR/"
data_dir_1='crop_224_resize_224'
data_dir_2='NG_resized'
epoch=49
lr=0.02
warm_epoch=3
warm_lr=0.1
post_epoch=49
cuda=0
expname="EM00_"
seed=55
resumes=("./experiments/output/SymmNets_NG06_SS07/${expname}EMR_r20200204_230222_NG06_elizabeth_162r20200201_225358_SS07_Middleton_17_SymmNetsV1_SourceOnlyclosed/ckpt_last.resume" \
"./experiments/output/SymmNets_NG06_SS09/${expname}EMR_r20200204_230222_NG06_elizabeth_162r20200203_033907_SS09_Middleton_10_SymmNetsV1_SourceOnlyclosed/ckpt_last.resume" \
"./experiments/output/SymmNets_SS07_NG06/${expname}EMR_r20200201_225358_SS07_Middleton_172r20200204_230222_NG06_elizabeth_16_SymmNetsV1_SourceOnlyclosed/ckpt_last.resume" \
"./experiments/output/SymmNets_SS07_SS09/${expname}EMR_r20200201_225358_SS07_Middleton_172r20200203_033907_SS09_Middleton_10_SymmNetsV1_SourceOnlyclosed/ckpt_last.resume" \
"./experiments/output/SymmNets_SS09_NG06/${expname}EMR_r20200203_033907_SS09_Middleton_102r20200204_230222_NG06_elizabeth_16_SymmNetsV1_SourceOnlyclosed/ckpt_last.resume" \
"./experiments/output/SymmNets_SS09_SS07/${expname}EMR_r20200203_033907_SS09_Middleton_102r20200201_225358_SS07_Middleton_17_SymmNetsV1_SourceOnlyclosed/ckpt_last.resume")
resumes_none=("./experiments/output/SymmNets_NG06_SS07/${expname}EMR_r20200204_230222_NG06_elizabeth_162r20200201_225358_SS07_Middleton_17_SymmNetsV1_Noneclosed/ckpt_last.resume" \
"./experiments/output/SymmNets_NG06_SS09/${expname}EMR_r20200204_230222_NG06_elizabeth_162r20200203_033907_SS09_Middleton_10_SymmNetsV1_Noneclosed/ckpt_last.resume" \
"./experiments/output/SymmNets_SS07_NG06/${expname}EMR_r20200201_225358_SS07_Middleton_172r20200204_230222_NG06_elizabeth_16_SymmNetsV1_Noneclosed/ckpt_last.resume" \
"./experiments/output/SymmNets_SS07_SS09/${expname}EMR_r20200201_225358_SS07_Middleton_172r20200203_033907_SS09_Middleton_10_SymmNetsV1_Noneclosed/ckpt_last.resume" \
"./experiments/output/SymmNets_SS09_NG06/${expname}EMR_r20200203_033907_SS09_Middleton_102r20200204_230222_NG06_elizabeth_16_SymmNetsV1_Noneclosed/ckpt_last.resume" \
"./experiments/output/SymmNets_SS09_SS07/${expname}EMR_r20200203_033907_SS09_Middleton_102r20200201_225358_SS07_Middleton_17_SymmNetsV1_Noneclosed/ckpt_last.resume")
yamls=('EMR_train_NG06_SS07_cfg.yaml' 'EMR_train_NG06_SS09_cfg.yaml' 'EMR_train_SS07_NG06_cfg.yaml' 'EMR_train_SS07_SS09_cfg.yaml' 'EMR_train_SS09_NG06_cfg.yaml' 'EMR_train_SS09_SS07_cfg.yaml')

for j in 0 1 2 3 4 5
do
  yaml="${yamls[$j]}"
  resume="${resumes[$j]}"
  resume_none="${resumes_none[$j]}"
  echo $resume $yaml
    for data_dir in 'crop_224_resize_224' 'NG_resized'
    do
      echo $lr $cbp $distance_type $data_dir
      # NO SCALING
      # Nothing
      CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'SourceOnly' --method SymmNetsV1 --cfg ./experiments/configs/EMR/${yaml} \
      --set DATASET.DATAROOT ${data_location}${data_dir} TRAIN.BASE_LR $lr MODEL.BP False SEED $seed TRAIN.MAX_EPOCH ${epoch} \
      INV.ALPHA 10.0
      #  - with Symmnet (NO BP)
      CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'None' --method SymmNetsV1 --cfg ./experiments/configs/EMR/${yaml} \
      --set DATASET.DATAROOT ${data_location}${data_dir} TRAIN.BASE_LR $lr MODEL.BP False SEED $seed TRAIN.MAX_EPOCH ${epoch} \
      INV.ALPHA 10.0
      # With BP & no SymmNet - warmup
      CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'SourceOnly' --method SymmNetsV1 --cfg ./experiments/configs/EMR/${yaml} \
      --set DATASET.DATAROOT ${data_location}${data_dir} MODEL.BP True SEED $seed TRAIN.MAX_EPOCH ${warm_epoch} \
      INV.ALPHA 10.0 BP.Q 4 BP.AB 64 BP.R 16 TRAIN.SAVING True TRAIN.LR_SCHEDULE 'fix' TRAIN.BASE_LR ${warm_lr}
      # With BP & no SymmNet - finish
      CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'SourceOnly' --method SymmNetsV1 --cfg ./experiments/configs/EMR/${yaml} \
      --set DATASET.DATAROOT ${data_location}${data_dir} TRAIN.BASE_LR $lr MODEL.BP True SEED $seed TRAIN.MAX_EPOCH ${post_epoch} \
      INV.ALPHA 10.0 BP.Q 4 BP.AB 64 BP.R 16 RESUME ${resume}
      # With BP and SymmNet - warmup
      CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'None' --method SymmNetsV1 --cfg ./experiments/configs/EMR/${yaml} \
      --set DATASET.DATAROOT ${data_location}${data_dir} MODEL.BP True SEED $seed TRAIN.MAX_EPOCH ${warm_epoch} \
      INV.ALPHA 10.0 BP.Q 4 BP.AB 64 BP.R 16 TRAIN.SAVING True TRAIN.LR_SCHEDULE 'fix' TRAIN.BASE_LR ${warm_lr}
      # With BP and SymmNet - finish
      CUDA_VISIBLE_DEVICES=$cuda python train.py --exp_name $expname --distance_type 'None' --method SymmNetsV1 --cfg ./experiments/configs/EMR/${yaml} \
      --set DATASET.DATAROOT ${data_location}${data_dir} TRAIN.BASE_LR $lr MODEL.BP True SEED $seed TRAIN.MAX_EPOCH ${post_epoch} \
      INV.ALPHA 10.0 BP.Q 4 BP.AB 64 BP.R 16 RESUME ${resume_none}
  done
done