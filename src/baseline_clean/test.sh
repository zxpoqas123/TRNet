#!/bin/bash
epochs=100
lr=1e-3
loss_type=CE
optimizer_type=Adam
batch_size=32
gpu_id=1

set -ue

# training the network

#feature='egemaps'
feature='FBank'
#feature='WavLM'
#feature='mfcc'
max_len=5
dataset_type=IEMOCAP_4
seed=22

noise_type=ESC50
for snr in 100 20 15 10 5 0
do
python test.py --seed $seed --device_number $gpu_id --feature $feature \
            --max_len $max_len --lr $lr --batch_size $batch_size --snr $snr --noise_type $noise_type \
            --optimizer_type $optimizer_type --dataset_type $dataset_type
done

noise_type=MUSAN
for snr in 100 20 15 10 5 0
do
python test.py --seed $seed --device_number $gpu_id --feature $feature \
            --max_len $max_len --lr $lr --batch_size $batch_size --snr $snr --noise_type $noise_type \
            --optimizer_type $optimizer_type --dataset_type $dataset_type
done