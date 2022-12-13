#!/usr/bin/env bash

if [[ "$1" == "TEMPONet" ]]; then
    echo TEMPONet testing
    python3 -u main.py --network TEMPONet --tcn_layers $2 --blocks $3 --dim_head $4 --heads $5 --depth $6 --patch_size1 $7 --patch_size2 $8 --patch_size3 $9 --ch_1 ${10} --ch_2 ${11} --ch_3 ${12} --subjects ${13} --pretrain ${14} --finetune ${15}
else
    echo ViT testing
    python3 -u main.py --network $1 --tcn_layers $2 --blocks $3 --dim_head $4 --heads $5 --depth $6 --patch_size1 $7 --patch_size2 $8 --patch_size3 $9 --ch_1 ${10} --ch_2 ${11} --ch_3 ${12} --subjects ${13} --pretrain ${14} --finetune ${15}
fi
