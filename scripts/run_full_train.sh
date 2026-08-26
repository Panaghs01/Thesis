#!/bin/bash
#simple head + augmentations + sgd + l1 reg

gpus=0
checkpoint_root=checkpoint_test

img_size=224
batch_size=64
lr=3e-4
lr_policy=warmup
max_epochs=2000
optimizer=adam
reset_lr=0

embedding_dim=128
num_embeddings=2048
commitment_cost=0.15
hiddens=256
residual_layers=6
residual_hiddens=512
vqvae_loss=mse

argloss=focal
focal_alpha=1.5
focal_gamma=2.0

lad_alpha=0.03
walk_steps=4

num_workers=8
project_name=lad_train_8
data_name=Fitzpatrick17k_balanced
train=strong_classifier
strong_classifier=base_resnet18

regularization=l2
lambda_reg=0.01

fine_tune_patience=20
fine_tune_delta=0.01

python new_main.py --gpu_ids ${gpus} --checkpoint_root ${checkpoint_root} \
    --img_size ${img_size} --batch_size ${batch_size} --lr ${lr} \
    --project_name ${project_name} \
    --data_name ${data_name} --train ${train} --max_epochs ${max_epochs} \
    --vqvae_num_embeddings ${num_embeddings} --num_workers ${num_workers} --vqvae_embedding_dim ${embedding_dim} --vqvae_commitment_cost ${commitment_cost}\
    --vqvae_hiddens ${hiddens} --vqvae_residual_hiddens ${residual_hiddens} --vqvae_residual_layers ${residual_layers} --vqvae_residual_hiddens ${residual_hiddens}\
    --vqvae_loss ${vqvae_loss} --optimizer ${optimizer} --reset_lr ${reset_lr} --lr_policy ${lr_policy} --lad_alpha ${lad_alpha} --walk_steps ${walk_steps}\
    --fine_tune_patience ${fine_tune_patience} --fine_tune_delta ${fine_tune_delta} --regularization ${regularization} --loss ${argloss}\
     --strong_classifier ${strong_classifier} --focal_alpha ${focal_alpha} --focal_gamma ${focal_gamma}

exit