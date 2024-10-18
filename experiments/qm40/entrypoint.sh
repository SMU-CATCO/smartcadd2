#!/bin/bash

cd experiments/qm40

python train_2d_models.py --dim 64 --batch_size 128 --target 0 --data_dir /data --save_dir /artifacts