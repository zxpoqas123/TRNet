#!/bin/bash
set -ue

gpu_id=1
max_len=5
noise_type=MUSAN
dataset_type=IEMOCAP_4

seed=2
for snr in 100 20
do
python generate_data_loso.py --device_number $gpu_id --max_len $max_len --snr $snr --noise_type $noise_type \
                     --dataset_type $dataset_type --seed $seed --do_se
done