import os, sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
import numpy as np
from pickle import dump, load
from time import time
import json 

from utils.db6 import DB6MultiSession
from utils.utils import SuperSet
from utils.model import ViT as ViT
from utils.model import TEMPONet as TEMPONet
from utils.download_DB6 import download_file
from utils.utils import get_loss_preds
from utils.train import train 
from utils.configs import configs_pretrain, configs_finetune, configs_finetune_nopretrain
import argparse



configs = list(ParameterGrid({k: (v if isinstance(v, list) else [v]) for k, v in configs_finetune_nopretrain.items()}))
configs = sorted(configs, key=lambda x: (x["subjects"], x["sessions"]) )
configs_chunks_finetune = []
dataset_combinations = set(map(lambda x: (x["subjects"], x["sessions"]), configs))
if len(dataset_combinations) == 1:
    for indices in np.array_split(np.arange(len(configs)), PROCESSES):
        if len(indices) > 0: # 
            configs_chunks_finetune.append(configs[indices[0]:indices[-1]+1])
else:
    prev_dataset_combination, new_chunk = None, None
    for config in configs:
        current_dataset_combination = (config["subjects"], config["sessions"])
        if current_dataset_combination != prev_dataset_combination:
            prev_dataset_combination = current_dataset_combination
            if new_chunk is not None:
                configs_chunks_finetune.append(new_chunk)
            new_chunk = []
        new_chunk.append(config)
    configs_chunks_finetune.append(new_chunk)
configs_chunks_idx_finetune = list(range(len(configs_chunks_finetune)))
configs = configs_chunks_finetune[0]
configs[0]["depth"] = 1
configs[0]["heads"] = 8
configs[0]["dim_head"] = 32
configs[0]["ch_1"] = 14
configs[0]["ch_2"] = 16
configs[0]["ch_3"] = 0
configs[0]["patch_size1"] = (1, 5)
configs[0]["patch_size2"] = (1, 3)
configs[0]["patch_size3"] = (1, 0)
configs[0]["tcn_layers"] = 2

# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(14, 1, 1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        # x = self.dequant(x)
        return x

# create a model instance
model_fp32 = ViT(**configs[0])
# model_fp32 = M()

# model must be set to eval mode for static quantization logic to work
model_fp32.eval()

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or assymetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Fuse the activations to preceding layers, where applicable.
# This needs to be done manually depending on the model architecture.
# Common fusions include `conv + relu` and `conv + batchnorm + relu`
# model_fp32 = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare(model_fp32)

# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
input_fp32 = torch.randn(1, 14, 1, 300)
model_fp32_prepared(input_fp32)

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
model_int8 = torch.quantization.convert(model_fp32_prepared)

# run the model, relevant calculations will happen in int8
res = model_int8(input_fp32)
print(res)
import pdb;pdb.set_trace()