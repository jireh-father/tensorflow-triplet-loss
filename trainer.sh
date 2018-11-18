#!/bin/bash
nohup python -u trainer.py --data_dir=/home/ubuntu/data/cafe24product/ --save_dir=experiments/cafe24product_pnasnet_large/ --save_epochs=3 --batch_size=128 --keep_checkpoint_max=40 --num_epochs=100 --gpu_no=0 --learning_rate=0.01 --model_name=pnasnet_large > train.log &
sudo shutdown now