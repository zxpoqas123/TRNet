## TRNet
This repository contains the implementation of the paper "TRNet: Two-level Refinement Network leveraging Speech Enhancement for Noise Robust Speech Emotion Recognition".

### Noise contaminated data preparation:
cd data_gen
bash pre_loso.sh

### Source code path:
cd src/baseline_clean: code for baseline-c in the paper
cd src/baseline_noise: code for baseline-n in the paper
cd src/baseline_enhanced: code for baseline-e in the paper
cd src/baseline_TRNet_wo_high: code for TRNet w/o L_{high} in the paper
cd src/baseline_TRNet: code for TRNet and TRNet w/o L_{low} in the paper

### Model training:
bash run.sh $gpu_id $model $hidden $seed $missing_rate $train_dynamic

### Model evaluation:
bash test.sh $gpu_id $model $hidden $seed $missing_rate $train_dynamic
