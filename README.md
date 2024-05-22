## TRNet
This repository contains the implementation of the paper "TRNet: Two-level Refinement Network leveraging Speech Enhancement for Noise Robust Speech Emotion Recognition".

### Noise contaminated data preparation:
cd data_gen
bash pre_loso.sh

### Source code path:
cd src/baseline_clean: code for baseline-c in the paper

cd src/baseline_noise: code for baseline-n in the paper

cd src/baseline_enhanced: code for baseline-e in the paper

cd src/baseline_TRNet_wo_high: code for TRNet w/o $L_{high}$ in the paper

cd src/baseline_TRNet: code for TRNet (i.e., alpha=0.5 and beta=0.5) and TRNet w/o $L_{low}$ (i.e., alpha=0.0 and beta=0.5) in the paper

### Model training:
bash run.sh

### Model evaluation:
bash test.sh

### Configuration statement:

gpu_id=0 (indicating which gpu to use) 

noise_type=ESC50 (indicating the noise type for speech contamination in training or evaluation)

snr=20 (indicating the signal-to-noise ratio in evaluation)

seed=2021 (indicating the random seed)

alpha=0.5 (indicating the importance of $L_{low}$)

beta=0.5 (indicating the importance of $L_{high}$)
