#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.db6 import DB6MultiSession
from pickle import load

import os


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[45]:


# MODEL

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self._init_parameters(dim, hidden_dim)

    def _init_parameters(self, dim, hidden_dim):
        bound1 = 1 / (dim ** .5)
        bound2 = 1 / (hidden_dim ** .5)
        nn.init.uniform_(self.net[0].weight, -bound1, bound1)
        nn.init.uniform_(self.net[0].bias, -bound1, bound1)
        nn.init.uniform_(self.net[3].weight, -bound2, bound2)
        nn.init.uniform_(self.net[0].bias, -bound2, bound2)

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self._init_parameters(dim, inner_dim)

    def _init_parameters(self, dim, inner_dim):
        bound = 1 / (dim ** .5)
        nn.init.uniform_(self.to_q.weight, -bound, bound)
        nn.init.uniform_(self.to_k.weight, -bound, bound)
        nn.init.uniform_(self.to_v.weight, -bound, bound)

        bound = 1 / (inner_dim ** .5)
        nn.init.uniform_(self.to_out[0].weight, -bound, bound)
        nn.init.uniform_(self.to_out[0].bias, -bound, bound)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = q.reshape(b, n, h, -1).permute(0, 2, 1, 3)
        k = k.reshape(b, n, h, -1).permute(0, 2, 1, 3)
        v = v.reshape(b, n, h, -1).permute(0, 2, 1, 3)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)

        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, window_size=(14, 300), patch_length=10, num_classes=8, dim=64, depth=1, heads=8, mlp_dim=128, pool='cls', dim_head=32, dropout=.2, emb_dropout=0., use_cls_token=True):
        super().__init__()

        channels, window_length = window_size
        num_patches = (window_length // patch_length)
        patch_dim = channels * patch_length
        self.patch_dim = patch_dim

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        #self.patch_conv = nn.Conv1d(in_channels=channels, out_channels=dim, kernel_size=patch_length, stride=patch_length, padding=0, bias=True)
        self.to_patch = nn.Linear(patch_dim, dim)

        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.empty(1, num_patches, dim))

        self.cls_token = nn.Parameter(torch.empty(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self._init_parameters(patch_dim)

    def _init_parameters(self, patch_dim):
        bound = 1 / (patch_dim ** .5)
        nn.init.uniform_(self.to_patch.weight, -bound, bound)
        nn.init.uniform_(self.to_patch.bias, -bound, bound)
        nn.init.zeros_(self.pos_embedding)
        nn.init.zeros_(self.mlp_head[1].weight)
        nn.init.zeros_(self.mlp_head[1].bias)

    def forward(self, x):
        #x = self.patch_conv(x).flatten(2).transpose(-2, -1)
        b = x.shape[0]
        x = self.to_patch(x.permute(0, 2, 1).reshape(b, -1, self.patch_dim))
        
        n = x.shape[1]

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
        else :
            x += self.pos_embedding

        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        return x


# In[22]:


def load_dataset(dataset_dir, subject):
    
    all_other_subjects = ','.join([str(s) for s in range(1, 11) if s != subject])
    minmax_picklename = f'./minmax/ds_minmax_sessions=5subjects={all_other_subjects}.pickle'
    minmax = load(open(minmax_picklename, 'rb'))
    
    test_ds = DB6MultiSession(folder=os.path.expanduser(dataset_dir), 
                              subjects=[subject], sessions=list(range(5, 10)), 
                              minmax=minmax, n_classes='7+1', steady=True).to(device)
    
    return test_ds


# In[23]:


def load_model(subject, training_fold):
    net = ViT()
    net.to(device)
    net.eval()
    net.load_state_dict((torch.load(f"checkpoints/vit_subject{subject}_fold{training_fold}.pth")))
    return net


# In[24]:


@torch.no_grad()
def get_loss_preds(net, criterion, loader):
    y_pred, y_true = [], []
    loss = 0
    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        outputs = net(X_batch)
        _, predicted = torch.max(outputs, 1)
        loss += criterion(outputs, Y_batch).item()

        y_pred.append(predicted.cpu())
        y_true.append(Y_batch.cpu())
        
    y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)
    loss /= len(loader)
    
    return loss, (y_pred, y_true)


# In[25]:


test_ds = load_dataset(dataset_dir='../../dataset_DB6', subject=5)
ds_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, pin_memory=False, drop_last=False)


# In[46]:


net = load_model(subject=5, training_fold=1)
_, (y_pred, y_true) = get_loss_preds(net, nn.CrossEntropyLoss(), ds_loader)
accuracy_fold1 = (y_pred == y_true).float().mean()

net = load_model(subject=5, training_fold=2)
_, (y_pred, y_true) = get_loss_preds(net, nn.CrossEntropyLoss(), ds_loader)
accuracy_fold2 = (y_pred == y_true).float().mean()

print(.5 * (accuracy_fold1 + accuracy_fold2))

