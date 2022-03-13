# -*- coding: utf-8 -*-
"""convit_db6_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16MWdEOGeCl8v6F1Ov6GDnDIwRIC5dDR6
"""


import os, sys
if os.path.expanduser('~/master-thesis') not in sys.path:
    sys.path.append(os.path.expanduser('~/master-thesis'))

if "original_print" not in locals():
    original_print = print
    def print(*args, **kwargs):
        kwargs['flush'] = True
        return original_print(*args, **kwargs)
    
# %load_ext autoreload
# %autoreload 2

PROCESSES = 5
device = 'cuda'
save_model_every_n = 1

from src.datasets.db6 import DB6MultiSession

import torch
@torch.no_grad()
def get_loss_preds(net, criterion, loader):
    y_pred, y_true = [], []
    loss = 0
    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        outputs = net(X_batch)
        #print(outputs)
        _, predicted = torch.max(outputs, 1)
        loss += criterion(outputs, Y_batch).item()

        y_pred.append(predicted)
        y_true.append(Y_batch)
        
    y_pred, y_true = torch.cat(y_pred).cpu(), torch.cat(y_true).cpu()
    loss /= len(loader)
    
    return loss, (y_pred, y_true)

from torch.utils.data import DataLoader

import torch
from torch import nn

import numpy as np

def train(net, net_name, ds, k, bootstrap, training_config, test_ds=None):

    assert bootstrap.split('_')[0] in {'no', '0', '1'}
    assert bootstrap.split('_')[1] in {'once', 'each-epoch'} if bootstrap.split('_')[0] != 'no' else True

    train_loader = None

    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    epochs, batch_size = training_config['epochs'], training_config['batch_size']
    optimizer = getattr(torch.optim, training_config['optim'])(net.parameters(), **training_config['optim_hparams'])
    scheduler = getattr(torch.optim.lr_scheduler, training_config['lr_scheduler'])(optimizer, **training_config['lr_scheduler_hparams'])

    print(epochs)
    print(optimizer)
    print(scheduler)

    torch.backends.cudnn.benchmark = True

    losses_accs = []

    for epoch in range(epochs):

        if 'each-epoch' in bootstrap or train_loader is None:
            if '0' in bootstrap:
                train_ds, val_ds = ds.split_0(total_folds=2, val_fold=k)
            elif '1' in bootstrap:
                train_ds, val_ds = ds.split_1(total_folds=2, val_fold=k)
            else:
                train_ds, val_ds = ds.split(total_folds=2, val_fold=k)
            
            if train_loader is None:
                print("Dataset lengths:", len(train_ds), len(val_ds))

            if train_ds[0][0].device.type == 'cpu':
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
                train_loader_ = DataLoader(train_ds, batch_size=1024, shuffle=False, pin_memory=True, drop_last=False)
                val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True, drop_last=False)
                if test_ds is not None:
                    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=True, drop_last=False)
            else :
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)
                train_loader_ = DataLoader(train_ds, batch_size=1024, shuffle=False, pin_memory=False, drop_last=False)
                val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=False, drop_last=False)
                if test_ds is not None:
                    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=False, drop_last=False)

        if epoch == 0:
            train_loss, (y_pred, y_true) = get_loss_preds(net, criterion, train_loader_)
            train_preds = y_pred.bincount(minlength=(y_true.max() + 1))
            train_acc = (y_true == y_pred).sum() / len(y_true)

            val_loss, (y_pred, y_true) = get_loss_preds(net, criterion, val_loader)
            val_preds = y_pred.bincount(minlength=(y_true.max() + 1))
            val_acc = (y_true == y_pred).sum() / len(y_true)

            test_loss, test_preds, test_acc = 0, '', 0
            if test_ds is not None:
                test_loss, (y_pred, y_true) = get_loss_preds(net, criterion, test_loader)
                test_preds = y_pred.bincount(minlength=(y_true.max() + 1))
                test_acc = (y_true == y_pred).sum() / len(y_true)

            with np.printoptions(precision=3, suppress=True):
                print(f"Train loss {train_loss:.3f}, Train acc {train_acc:.3f} {train_preds:},\n{' '* 20}Val loss {val_loss:.3f} Val acc {val_acc:.3f}, {val_preds},\n{' '* 40}Test loss {test_loss:.3f} Test acc {test_acc:.3f}, {test_preds}")


        net.train()
        print("Current LR:", optimizer.param_groups[0]['lr'])
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()

            outputs = net(X_batch)

            train_loss = criterion(outputs, Y_batch)
            train_loss.backward()

            #torch.nn.utils.clip_grad_norm_(net.parameters(), 1e-5)

            optimizer.step()        

        scheduler.step()

        net.eval()

        train_loss, (y_pred, y_true) = get_loss_preds(net, criterion, train_loader_)
        train_preds = y_pred.bincount(minlength=(y_true.max() + 1))
        train_acc = (y_true == y_pred).sum() / len(y_true)

        val_loss, (y_pred, y_true) = get_loss_preds(net, criterion, val_loader)
        val_preds = y_pred.bincount(minlength=(y_true.max() + 1))
        val_acc = (y_true == y_pred).sum() / len(y_true)

        test_loss, test_preds, test_acc = 0, '', 0
        if test_ds is not None:
            test_loss, (y_pred, y_true) = get_loss_preds(net, criterion, test_loader)
            test_preds = y_pred.bincount(minlength=(y_true.max() + 1))
            test_acc = (y_true == y_pred).sum() / len(y_true)

        with np.printoptions(precision=3, suppress=True):
            print(f"Epoch {epoch + 1:02d}, Train loss {train_loss:.3f}, Train acc {train_acc:.3f} {train_preds:},\n{' '* 20}Val loss {val_loss:.3f} Val acc {val_acc:.3f}, {val_preds},\n{' '* 40}Test loss {test_loss:.3f} Test acc {test_acc:.3f}, {test_preds}")
            losses_accs.append({'train_loss': float(train_loss), 'train_acc': float(train_acc), 'val_loss': float(val_loss), 'val_acc': float(val_acc), 'test_loss': float(test_loss), 'test_acc': float(test_acc)})

        if ((epoch + 1) % save_model_every_n) == 0:
            torch.save(net.state_dict(), f"{net_name}_epoch{epoch+1}.pth")

        #print(os.popen('free -h').read())
        #print(os.popen('uptime').read())

    return losses_accs

from pickle import dump, load

ds_cache = {'ds_config': None}

def main_1test(chunk_idx):
    configs = configs_chunks[chunk_idx]

    subjects = configs[0]['subjects']
    n_sessions = configs[0]['sessions']
    
    minmax = True
    if configs[0]["pretrained"] is not None:
        minmax_picklename = f'ds_minmax_sessions={n_sessions}subjects=1,2,3,4,5,6,7,8,9,10.pickle'
        print("Loading minmax from", minmax_picklename)
        minmax = load(open(minmax_picklename, 'rb'))

    ds_config = dict(folder=os.path.expanduser('~/DB6'), subjects=subjects, sessions=list(range(n_sessions)), minmax=minmax, n_classes='7+1', steady=True, image_like_shape=True)
    if ds_cache['ds_config'] != ds_config:
        ds = DB6MultiSession(folder=os.path.expanduser('~/DB6'), subjects=subjects, sessions=list(range(n_sessions)), minmax=minmax, n_classes='7+1', steady=True, image_like_shape=True).to('cuda')
        test_ds = DB6MultiSession(folder=os.path.expanduser('~/DB6'), subjects=subjects, sessions=[n_sessions], minmax=(ds.X_min, ds.X_max), n_classes='7+1', steady=True, image_like_shape=True).to('cuda')
        if is_notebook():
            ds_cache['ds'] = ds
            ds_cache['test_ds'] = test_ds
            ds_cache['ds_config'] = ds_config

    if is_notebook():
        ds, test_ds, ds_config = ds_cache['ds'], ds_cache['test_ds'], ds_cache['ds_config']

    minmax_picklename = f'ds_minmax_sessions={n_sessions}subjects={ ",".join(map(str, subjects)) }.pickle'
    if not os.path.isfile(minmax_picklename):
        dump([ds.X_min, ds.X_max], open(minmax_picklename, 'wb'))

    losses_accs_ = []
    n_params_ = []
    for i, config in enumerate(configs, start=1):

        config["chunk_idx"] = chunk_idx
        config["chunk_i"] = i
        config["conv_layers"] = [(0,), (0,1,), (0,1,2)][config["depth"] - 2]
        net = VisionTransformer(norm_layer=partial(nn.LayerNorm, eps=1e-6), **config)
        if config['pretrained'] is not None:
            net.load_state_dict(torch.load(config['pretrained']))
            print("Loaded checkpoint", config['pretrained'])
    
        n_params = sum([param.nelement() for param in net.parameters()])
        n_params_.append(n_params)
        
        print(f"Run {i}/{len(configs)}")
        print(config)
        print(f"Params: {n_params / 10**6:.3f}M")
        losses_accs = train(net=net, net_name=f"{name_prefix}_{chunk_idx}_{i}", ds=ds, k=0, bootstrap='no', training_config=config['training_config'], test_ds=test_ds)
        losses_accs_.append(losses_accs)
        print()

    return [configs, n_params_, losses_accs_]

from src.datasets.db6 import DB6MultiSession

def main_train_subject(chunk_idx):
    configs = configs_chunks[chunk_idx]

    subject = configs[0]['subjects']
    n_sessions = configs[0]['sessions']
    train_sessions = list(range(configs[0]['sessions']))
    test_sessions = list(range(configs[0]['sessions'], 10))
    #test_sessions = list(range(configs[0]['sessions'], 2))

    steady=True
    n_classes='7+1'
    bootstrap='no'

    print("Training subject", subject)
    print("Steady", steady)
    print("Bootstrap", bootstrap)

    minmax = True
    if configs[0]["pretrained"] is not None:
        minmax_picklename = f'ds_minmax_sessions={n_sessions}subjects=1,2,3,4,5,6,7,8,9,10.pickle'
        print("Loading minmax from", minmax_picklename)
        minmax = load(open(minmax_picklename, 'rb'))

    ds = DB6MultiSession(folder=os.path.expanduser('~/DB6'), subjects=[subject], sessions=train_sessions, steady=steady, n_classes=n_classes, minmax=minmax, image_like_shape=True).to('cuda')
    test_ds_5 = DB6MultiSession(folder=os.path.expanduser('~/DB6'), subjects=[subject], sessions=test_sessions, steady=steady, n_classes=n_classes, minmax=(ds.X_min, ds.X_max), image_like_shape=True).to('cuda')

    test_datasets_steady = [DB6MultiSession(folder=os.path.expanduser('~/DB6'), subjects=[subject], sessions=[i], 
                                            steady=True, n_classes=n_classes, minmax=(ds.X_min, ds.X_max), image_like_shape=True) \
                            for i in test_sessions]
    test_datasets = [DB6MultiSession(folder=os.path.expanduser('~/DB6'), subjects=[subject], sessions=[i], 
                                     steady=False, n_classes=n_classes, minmax=(ds.X_min, ds.X_max), image_like_shape=True) \
                            for i in test_sessions]
    
    results_ = []

    for i, config in enumerate(configs, start=1):
        results = {}
        results['subject'] = subject
        results['steady'] = steady
        results['n_classes'] = n_classes
        results['bootstrap'] = bootstrap
        results['train_sessions'] = train_sessions
        results['test_sessions'] = test_sessions
        for k in [0, 1]:
            result = {}

            net = VisionTransformer(norm_layer=partial(nn.LayerNorm, eps=1e-6), **config)

            if config['pretrained'] is not None:
                net.load_state_dict(torch.load(config['pretrained']))
                print("Loaded checkpoint", config['pretrained'])

            losses_accs = train(net=net, net_name=f"{name_prefix}_{chunk_idx}_{i}", ds=ds, k=k, bootstrap=bootstrap, training_config=config['training_config'], test_ds=test_ds_5)
            result['losses_accs'] = losses_accs

            criterion = nn.CrossEntropyLoss()

            test_losses, y_preds, y_trues = [], [], []
            for test_ds in test_datasets_steady:
                test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=False, drop_last=False)
                test_loss, (y_pred, y_true) = get_loss_preds(net, criterion, test_loader)
                test_losses.append(test_loss)
                y_preds.append(y_pred.cpu())
                y_trues.append(y_true.cpu())
            result['test_losses_steady'] = test_losses
            result['y_preds_steady'] = y_preds
            result['y_trues_steady'] = y_trues

            test_losses, y_preds, y_trues = [], [], []
            for test_ds in test_datasets:
                test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=False, drop_last=False)
                test_loss, (y_pred, y_true) = get_loss_preds(net, criterion, test_loader)
                test_losses.append(test_loss)
                y_preds.append(y_pred.cpu())
                y_trues.append(y_true.cpu())
            result['test_losses'] = test_losses
            result['y_preds'] = y_preds
            result['y_trues'] = y_trues
        
            results[f'val-fold_{k}'] = result
        
        results_.append(results)

    return [configs, results_]

from sklearn.model_selection import ParameterGrid

from time import time
#name_prefix = f"vit{time():.0f}"
name_prefix = f"convit"

patch_size = 20
configs_ = {
    "img_size": (1, 300),
    "patch_size": (1, 10),
    "in_chans": 14,
    "num_classes": 8, 
    
    "depth": [2, 3, 4],
    "num_heads": 3,
    "embed_dim": 32,
    "mlp_ratio": 2.,

    "drop_rate": 0,
    "conv_layers": None,

    "sessions": 5,
    "subjects": (1,2,3,4,5,6,7,8,9,10),

    "pretrained": None,

    "training_config": [
        #{
        # "epochs": 30,
        # "batch_size": 64,
        # "optim": "Adam",
        # "optim_hparams": {"lr": 1e-3, "betas": (.9, .999), "weight_decay": 0},
        # "lr_scheduler": "StepLR",
        # "lr_scheduler_hparams": {"step_size": 10, "gamma": .1},
        #},
        {
         "epochs": 75,
         "batch_size": 64,
         "optim": "AdamW",
         "optim_hparams": {"lr": 0, "betas": (.9, .999), "weight_decay": 0.01},
         "lr_scheduler": "CyclicLR",
         "lr_scheduler_hparams": {"base_lr": 1e-7, "max_lr": 1e-3, "step_size_up": 50, "step_size_down": None, "mode": 'triangular', "cycle_momentum": False},
        },
        #{
        # "epochs": 10,
        # "batch_size": 64,
        # "optim": "Adam",
        # "optim_hparams": {"lr": 0, "betas": (.9, .999), "weight_decay": 0},
        # "lr_scheduler": "CyclicLR",
        # "lr_scheduler_hparams": {"base_lr": 1e-7, "max_lr": 1e-3, "step_size_up": 150, "step_size_down": None, "mode": 'triangular', "cycle_momentum": False},
        #},
    ]
}

configs = list(ParameterGrid({k: (v if isinstance(v, list) else [v]) for k, v in configs_.items()}))
configs = sorted(configs, key=lambda x: (x["subjects"], x["sessions"]) )

import numpy as np
configs_chunks = []

dataset_combinations = set(map(lambda x: (x["subjects"], x["sessions"]), configs))
if len(dataset_combinations) == 1:
    # Tutte le config usano stesso dataset -> N chunk dove N=PROCESSES
    for indices in np.array_split(np.arange(len(configs)), PROCESSES):
        if len(indices) > 0: # può essere 0 se ho per es 1 job e 4 processi
            configs_chunks.append(configs[indices[0]:indices[-1]+1])
else:
    # Diversi dataset -> N chunk dove N=combinazioni dei dataset
    prev_dataset_combination, new_chunk = None, None
    for config in configs:
        current_dataset_combination = (config["subjects"], config["sessions"])
        if current_dataset_combination != prev_dataset_combination:
            prev_dataset_combination = current_dataset_combination
            if new_chunk is not None:
                configs_chunks.append(new_chunk)
            new_chunk = []
        new_chunk.append(config)
    configs_chunks.append(new_chunk)

configs_chunks_idx = list(range(len(configs_chunks)))

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

'''These modules are adapted from those of timm, see
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvAtt(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.proj(x)

class GPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)       
        self.v = nn.Linear(dim, dim, bias=qkv_bias)       
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.apply(self._init_weights)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1)!=N:
            self.get_rel_indices(N)

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape        
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1,-1)
        pos_score = self.pos_proj(pos_score).permute(0,3,1,2) 
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1,-1,1,1)
        attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map = False):

        attn_map = self.get_attention(x).mean(0) # average over batch
        distances = self.rel_indices.squeeze()[:,:,-1]**.5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist
    
    def local_init(self, locality_strength=1.):
        
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1 #max(1,1/locality_strength**.5)
        
        kernel_size = int(self.num_heads**.5)
        center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1+kernel_size*h2
                self.pos_proj.weight.data[position,2] = -1
                self.pos_proj.weight.data[position,1] = 2*(h1-center)*locality_distance
                self.pos_proj.weight.data[position,0] = 2*(h2-center)*locality_distance
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self, num_patches):
        #img_size = int(num_patches**.5)
        #rel_indices   = torch.zeros(1, num_patches, num_patches, 3)
        #ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        #indx = ind.repeat(img_size,img_size)
        #indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        #indd = indx**2 + indy**2
        #rel_indices[:,:,:,2] = indd.unsqueeze(0)
        #rel_indices[:,:,:,1] = indy.unsqueeze(0)
        #rel_indices[:,:,:,0] = indx.unsqueeze(0)
        #device = self.qk.weight.device
        #self.rel_indices = rel_indices.to(device)

        img_size = (1, 300)
        patch_size = (1, 300 // num_patches)

        num_patches_x = img_size[1] // patch_size[1]
        num_patches_y = img_size[0] // patch_size[0]

        rel_indices = torch.zeros(1, num_patches_x * num_patches_y, num_patches_x * num_patches_y, 3)
        ind = torch.arange(num_patches_x).view(1,-1) - torch.arange(num_patches_x).view(-1, 1)
        indx = ind.repeat(num_patches_y, 1).repeat(1, num_patches_y)
        ind = torch.arange(num_patches_y).view(1,-1) - torch.arange(num_patches_y).view(-1, 1)
        indy = ind.repeat_interleave(num_patches_y, dim=0).repeat_interleave(num_patches_y, dim=1)
        indd = indx**2 + indy**2
        rel_indices[:,:,:,2] = indd.unsqueeze(0)
        rel_indices[:,:,:,1] = indy.unsqueeze(0)
        rel_indices[:,:,:,0] = indx.unsqueeze(0)

        device = self.qk.weight.device
        self.rel_indices = rel_indices.to(device)

 
class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N**.5)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indd = indx**2 + indy**2
        distances = indd**.5
        distances = distances.to('cuda')

        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N
        
        if return_map:
            return dist, attn_map
        else:
            return dist

            
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):

    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_conv=True, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #self.use_gpsa = use_gpsa
        if use_conv:
            #self.attn = GPSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
            self.attn = ConvAtt(dim, kernel_size=num_heads)
        else:
            self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding, from timm
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.apply(self._init_weights)
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding, from timm
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=48, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, global_pool=None,
                 conv_layers=(-1,), locality_strength=1., use_pos_embed=True, **extra):
        super().__init__()

        print("ignored params", extra)

        embed_dim *= num_heads
        self.num_classes = num_classes
        #self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        """self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=True,
                locality_strength=locality_strength)
            if i<local_up_to_layer else
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=False)
            for i in range(depth)])"""
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_conv=i in conv_layers,
                )
            for i in range(depth)
            ])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for u,blk in enumerate(self.blocks):
            if u == 0: #self.local_up_to_layer :
                x = torch.cat((cls_tokens, x), dim=1)
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
import torch
import torch.nn as nn
from functools import partial

from timm.models.efficientnet import EfficientNet
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model

@register_model
def convit_tiny(pretrained=False, **kwargs):
    num_heads = 4
    kwargs['embed_dim'] *= num_heads
    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/convit/convit_tiny.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model

@register_model
def convit_small(pretrained=False, **kwargs):
    num_heads = 9
    kwargs['embed_dim'] *= num_heads
    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/convit/convit_small.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model

@register_model
def convit_base(pretrained=False, **kwargs):
    num_heads = 16
    kwargs['embed_dim'] *= num_heads
    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/convit/convit_base.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model

import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from pickle import dump
from time import time

def is_notebook():
    return True #'get_ipython' in globals()

def extend_results(results, result):
    if results is None:
        return result
    
    for i in range(len(results)): # n campi di results (config, ecc)
        results[i].extend(result[i])
    
    return results

main = main_1test
#main = main_train_subject

print(__name__)

if __name__ == '__main__':

    results = None
    
    if is_notebook():
        for idx in configs_chunks_idx:
            results = extend_results(results, main(idx))
    else:
        with multiprocessing.Pool(PROCESSES) as pool:
            for result in pool.imap_unordered(main, configs_chunks_idx):
                results = extend_results(results, result)

    pickle_name = f'results_{time():.0f}.pickle'
    dump(results, open(pickle_name, 'wb'))
    print("Saved", pickle_name)
