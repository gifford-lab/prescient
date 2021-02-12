# example script for training 5 seeds of 2 layer 400 unit models 
# on Veres et al. dataset with estimated proliferation rates

python=/data/gl/g5/yhtgrace/env/miniconda3/envs/cuda10.1/bin/python
train_dt=0.1
train_sd=0.1
train_tau=1e-6
train_batch=0.1
train_clip=0.1 
train_lr=0.001
pretrain_epochs=500
train_epochs=2500
k_dim=400 
layers=2
save=100

data_dir=./data/Veres2019
out_dir=./experiments/veres-fate
weight_path=./data/Veres2019_growth-kegg.pt

device=0
seed=1
screen -dm bash -c "cd ../../; $python src/veres.py --task fate --train --data_dir $data_dir --out_dir $out_dir --train_batch $train_batch --train_clip $train_clip --train_lr $train_lr --train_sd $train_sd --train_dt $train_dt --train_tau $train_tau --pretrain_epochs $pretrain_epochs --train_epochs $train_epochs --k_dim $k_dim --layers $layers --save $save --seed $seed --device $device --weight_path $weight_path"

device=1
seed=2
screen -dm bash -c "cd ../../; $python src/veres.py --task fate --train --data_dir $data_dir --out_dir $out_dir --train_batch $train_batch --train_clip $train_clip --train_lr $train_lr --train_sd $train_sd --train_dt $train_dt --train_tau $train_tau --pretrain_epochs $pretrain_epochs --train_epochs $train_epochs --k_dim $k_dim --layers $layers --save $save --seed $seed --device $device --weight_path $weight_path"

device=2
seed=3
screen -dm bash -c "cd ../../; $python src/veres.py --task fate --train --data_dir $data_dir --out_dir $out_dir --train_batch $train_batch --train_clip $train_clip --train_lr $train_lr --train_sd $train_sd --train_dt $train_dt --train_tau $train_tau --pretrain_epochs $pretrain_epochs --train_epochs $train_epochs --k_dim $k_dim --layers $layers --save $save --seed $seed --device $device --weight_path $weight_path"

device=3
seed=4
screen -dm bash -c "cd ../../; $python src/veres.py --task fate --train --data_dir $data_dir --out_dir $out_dir --train_batch $train_batch --train_clip $train_clip --train_lr $train_lr --train_sd $train_sd --train_dt $train_dt --train_tau $train_tau --pretrain_epochs $pretrain_epochs --train_epochs $train_epochs --k_dim $k_dim --layers $layers --save $save --seed $seed --device $device --weight_path $weight_path"

device=4
seed=5
screen -dm bash -c "cd ../../; $python src/veres.py --task fate --train --data_dir $data_dir --out_dir $out_dir --train_batch $train_batch --train_clip $train_clip --train_lr $train_lr --train_sd $train_sd --train_dt $train_dt --train_tau $train_tau --pretrain_epochs $pretrain_epochs --train_epochs $train_epochs --k_dim $k_dim --layers $layers --save $save --seed $seed --device $device --weight_path $weight_path"
