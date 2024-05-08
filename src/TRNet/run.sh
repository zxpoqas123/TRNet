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
#feature='spectrogram'
max_len=5
noise_type=ESC50
dataset_type=IEMOCAP_4
snr=20

seed=2
alpha=0.0
beta=0.5

for fold in {0..9}
do
    store_root=./seed_$seed/alpha$alpha+beta$beta+$dataset_type+$max_len\s+$feature+lr$lr+batch_size$batch_size+$loss_type+$optimizer_type/
    echo "============training seed $seed============"
    python ./train.py \
        --seed $seed \
        --snr $snr \
        --max_len $max_len \
        --lr $lr \
        --fold $fold \
        --epochs $epochs \
        --root $store_root \
        --loss_type $loss_type \
        --batch_size $batch_size \
        --dataset_type $dataset_type \
        --optimizer_type $optimizer_type \
        --device_number $gpu_id \
        --feature $feature  \
        --noise_type $noise_type \
        --alpha $alpha \
        --beta $beta \
        --do_se 
done

noise_type=ESC50
for snr in 100 20 15 10 5 0
do
python test.py --seed $seed --device_number $gpu_id --feature $feature \
            --max_len $max_len --lr $lr --batch_size $batch_size --snr $snr --noise_type $noise_type \
            --optimizer_type $optimizer_type --dataset_type $dataset_type --alpha $alpha --beta $beta --do_se
done

noise_type=MUSAN
for snr in 100 20 15 10 5 0
do
python test.py --seed $seed --device_number $gpu_id --feature $feature \
            --max_len $max_len --lr $lr --batch_size $batch_size --snr $snr --noise_type $noise_type \
            --optimizer_type $optimizer_type --dataset_type $dataset_type --alpha $alpha --beta $beta --do_se
done