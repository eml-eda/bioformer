#!/usr/bin/env bash


python3 -u main_quantize.py --network $1 --tcn_layers $2 --blocks $3 --dim_head $4 --heads $5 --depth $6 --patch_size1 $7 --patch_size2 $8 --patch_size3 $9 --ch_1 ${10} --ch_2 ${11} --ch_3 ${12} --pretrain ${13} --finetune ${14}

