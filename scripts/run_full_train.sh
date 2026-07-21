#!/bin/bash

gpus=0
checkpoint_root=checkpoint_test

img_size=224
batch_size=32
lr=1e-4
lr_policy=plateau
max_epochs=110
optimizer=adam
reset_lr=0
class_weight=0.3

embedding_dim=128
num_embeddings=2048
commitment_cost=0.15
hiddens=256
residual_layers=6
residual_hiddens=512
vqvae_loss=mse

lad_alpha=0.1
walk_steps=20

num_workers=8
project_name=retrain_classifier
data_name=Fitzpatrick17k
train=strong_classifier

python new_main.py --gpu_ids ${gpus} --checkpoint_root ${checkpoint_root} \
    --img_size ${img_size} --batch_size ${batch_size} --lr ${lr} \
    --project_name ${project_name} \
    --data_name ${data_name} --train ${train} --max_epochs ${max_epochs} \
    --vqvae_num_embeddings ${num_embeddings} --num_workers ${num_workers} --vqvae_embedding_dim ${embedding_dim} --vqvae_commitment_cost ${commitment_cost}\
    --vqvae_hiddens ${hiddens} --vqvae_residual_hiddens ${residual_hiddens} --vqvae_residual_layers ${residual_layers} --vqvae_residual_hiddens ${residual_hiddens}\
    --vqvae_loss ${vqvae_loss} --optimizer ${optimizer} --reset_lr ${reset_lr} --lr_policy ${lr_policy} --lad_alpha ${lad_alpha} --walk_steps ${walk_steps}\
    --class_weight ${class_weight}
