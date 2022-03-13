# -*- coding: utf-8 -*-
"""vit_db6_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XZT0SosdQB_g1yhu_z9Yl4_jozFonAzn
"""

# Commented out IPython magic to ensure Python compatibility.
#! [ -f ~/.ssh/id_rsa_github ] || (echo "Downloading repo private key" && (mkdir -p ~/.ssh ; wget -qq https://www.dropbox.com/s/9vw58w194ozr3ov/id_rsa_master-thesis?dl=1 -O ~/.ssh/id_rsa_github ; chmod 400 ~/.ssh/id_rsa_github))
#! (grep --quiet --no-messages github ~/.ssh/config) || (echo "Adding GitHub public key" && echo "github.com ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAq2A7hRGmdnm9tUDbO9IDSwBK6TbQa+PXYPCPy6rbTrTtw7PHkccKrpp0yVhp5HdEIcKr6pLlVDBfOLX9QUsyCOV0wzfjIJNlGEYsdlLJizHhbn2mUjvSAHQqZETYP81eFzLQNnPHt4EVVUh7VfDESU84KezmD5QlWpXLmvU31/yMf+Se8xhHTvKSCZIFImWwoG6mbUoWf9nzpIoaSjB+weqqUUmpaaasXVal72J+UX2B+2RPW3RcT0eOzQgqlJL3RKrTJvdsjE3JEAvGq3lGHSZXy28G3skua2SmVi/w4yCE6gbODqnTWlg7+wC604ydGXA8VJiS5ap43JXiUFFAaQ==" >> ~/.ssh/known_hosts)
#! (grep --quiet --no-messages github ~/.ssh/config) || (echo "Adding repo private key" && echo "IdentityFile ~/.ssh/id_rsa_github" >> ~/.ssh/config)
#! [ -d ~/master-thesis ] || (echo "Cloning codebase" && cd ~ && git clone --quiet git@github.com:francibm97/master-thesis.git)

import os, sys
if os.path.expanduser('~/master-thesis') not in sys.path:
    sys.path.append(os.path.expanduser('~/master-thesis'))
    
# %load_ext autoreload
# %autoreload 2

if "original_print" not in locals():
    original_print = print
    def print(*args, **kwargs):
        kwargs['flush'] = True
        return original_print(*args, **kwargs)

#! (cd ~/master-thesis && git pull)

#! pip install scipy sklearn pandas

#! mkdir -p ~/DB6
#! cd ~/DB6 && wget -q --show-progress -r --no-parent --no-host-directories --cut-dirs=10 --accept 'S5_*' http://admin:nimda@lino1.francesco.pw/db6/

#! pip install einops

import torch

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

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

        #nn.init.xavier_uniform_(self.net[0].weight, gain=2 ** .5)
        #nn.init.normal_(self.net[0].bias, std=1e-6)
        #nn.init.xavier_uniform_(self.net[3].weight, gain=2 ** .5)
        #nn.init.normal_(self.net[3].bias, std=1e-6)
        #nn.init.kaiming_uniform_(self.net[0].weight, a=5**.5)
        #nn.init.kaiming_uniform_(self.net[0].weight, a=5**.5)
        bound1 = 1 / (dim ** .5)
        bound2 = 1 / (hidden_dim ** .5)
        nn.init.uniform_(self.net[0].weight, -bound1, bound1)
        nn.init.uniform_(self.net[0].bias, -bound1, bound1)
        nn.init.uniform_(self.net[3].weight, -bound2, bound2)
        nn.init.uniform_(self.net[0].bias, -bound2, bound2)

        #nn.init.xavier_normal_(self.net[0].weight, gain=2**-.5)
        #nn.init.normal_(self.net[0].bias, std=.1)
        #nn.init.xavier_normal_(self.net[3].weight, gain=2**-.5)
        #nn.init.normal_(self.net[3].bias, std=.1)

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
               
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        #nn.init.xavier_uniform_(self.to_qkv.weight)
        bound = 1 / (dim ** .5)
        nn.init.uniform_(self.to_qkv.weight, -bound, bound)
        #nn.init.xavier_normal_(self.to_qkv.weight, gain=2**-.5)
        #nn.init.normal_(self.to_qkv.weight, std=1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        #nn.init.xavier_uniform_(self.to_out[0].weight)
        #nn.init.normal_(self.to_out[0].bias, std=1e-6)
        bound = 1 / (inner_dim ** .5)
        nn.init.uniform_(self.to_out[0].weight, -bound, bound)
        nn.init.uniform_(self.to_out[0].bias, -bound, bound)

        #nn.init.zeros_(self.to_out[0].weight)
        #nn.init.zeros_(self.to_out[0].bias)

        #nn.init.xavier_normal_(self.to_out[0].weight, gain=1)
        #nn.init.normal_(self.to_out[0].bias, std=.05)


    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
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
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., use_cls_token=True, sessions="ignore"):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        bound = 1 / (patch_dim ** .5)
        nn.init.uniform_(self.to_patch_embedding[1].weight, -bound, bound)
        nn.init.uniform_(self.to_patch_embedding[1].bias, -bound, bound)
        #nn.init.xavier_uniform_(self.to_patch_embedding[1].weight)
        #nn.init.normal_(self.to_patch_embedding[1].bias, std=1e-6)
        #nn.init.xavier_normal_(self.to_patch_embedding[1].weight, gain=2**-.5)
        #nn.init.normal_(self.to_patch_embedding[1].bias, std=.1)

        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.empty(1, num_patches, dim))
        nn.init.normal_(self.pos_embedding, mean=0, std=.02)
        #nn.init.zeros_(self.pos_embedding)

        self.cls_token = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.zeros_(self.cls_token)
        
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        #nn.init.zeros_(self.mlp_head[1].weight)
        #nn.init.zeros_(self.mlp_head[1].bias)
        bound = 1 / (dim ** .5)
        nn.init.uniform_(self.mlp_head[1].weight, -bound, bound)
        nn.init.uniform_(self.mlp_head[1].bias, -bound, bound)
        #nn.init.xavier_normal_(self.mlp_head[1].weight, gain=1)
        #nn.init.normal_(self.mlp_head[1].weight, std=1)
        #nn.init.normal_(self.mlp_head[1].bias, std=1)

    def forward(self, img):
        
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
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

from torch.utils.data import DataLoader

import torch
from torch import nn

import numpy as np

def train(net, ds, k, bootstrap, test_ds, epochs):

    assert bootstrap.split('_')[0] in {'no', '0', '1'}
    assert bootstrap.split('_')[1] in {'once', 'each-epoch'} if bootstrap.split('_')[0] != 'no' else True

    train_loader = None

    net = net.to(device)
    #criterion = nn.CrossEntropyLoss(weight=torch.tensor([.1, 1, 1, 1, 1, 1, 1, 1]).to(device))
    criterion = nn.CrossEntropyLoss()
    lr0 = 1e-3 #5e-5
    wd = 0
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr0, momentum=0.9, dampening=0, weight_decay=wd) # weight_decay=5e-4
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr0, momentum=0.9, weight_decay=wd) # weight_decay=5e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=lr0, betas=(.9, .999), weight_decay=wd) # weight_decay=5e-4
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    #epochs = 100

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

            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
            train_loader_ = DataLoader(train_ds, batch_size=1024, shuffle=False, pin_memory=True, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True, drop_last=False)
            test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=True, drop_last=False)

        if epoch == 0:
            train_loss, (y_pred, y_true) = get_loss_preds(net, criterion, train_loader_)
            train_preds = y_pred.bincount(minlength=(y_true.max() + 1))
            train_acc = (y_true == y_pred).sum() / len(y_true)

            val_loss, (y_pred, y_true) = get_loss_preds(net, criterion, val_loader)
            val_preds = y_pred.bincount(minlength=(y_true.max() + 1))
            val_acc = (y_true == y_pred).sum() / len(y_true)

            test_loss, (y_pred, y_true) = get_loss_preds(net, criterion, test_loader)
            test_preds = y_pred.bincount(minlength=(y_true.max() + 1))
            test_acc = (y_true == y_pred).sum() / len(y_true)

            with np.printoptions(precision=3, suppress=True):
                print(f"Train loss {train_loss:.3f}, Train acc {train_acc:.3f} {train_preds:},\n{' '* 20}Val loss {val_loss:.3f} Val acc {val_acc:.3f}, {val_preds},\n{' '* 40}Test loss {test_loss:.3f} Test acc {test_acc:.3f}, {test_preds}")


        net.train()
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()

            outputs = net(X_batch)

            train_loss = criterion(outputs, Y_batch)
            train_loss.backward()

            optimizer.step()        

        scheduler.step()

        net.eval()

        train_loss, (y_pred, y_true) = get_loss_preds(net, criterion, train_loader_)
        train_preds = y_pred.bincount(minlength=(y_true.max() + 1))
        train_acc = (y_true == y_pred).sum() / len(y_true)

        val_loss, (y_pred, y_true) = get_loss_preds(net, criterion, val_loader)
        val_preds = y_pred.bincount(minlength=(y_true.max() + 1))
        val_acc = (y_true == y_pred).sum() / len(y_true)

        test_loss, (y_pred, y_true) = get_loss_preds(net, criterion, test_loader)
        test_preds = y_pred.bincount(minlength=(y_true.max() + 1))
        test_acc = (y_true == y_pred).sum() / len(y_true)

        with np.printoptions(precision=3, suppress=True):
            print(f"Epoch {epoch + 1:02d}, Train loss {train_loss:.3f}, Train acc {train_acc:.3f} {train_preds:},\n{' '* 20}Val loss {val_loss:.3f} Val acc {val_acc:.3f}, {val_preds},\n{' '* 40}Test loss {test_loss:.3f} Test acc {test_acc:.3f}, {test_preds}")
            losses_accs.append({'train_loss': float(train_loss), 'train_acc': float(train_acc), 'val_loss': float(val_loss), 'val_acc': float(val_acc), 'test_loss': float(test_loss), 'test_acc': float(test_acc)})

        #print(os.popen('free -h').read())
        #print(os.popen('uptime').read())

    return losses_accs

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

        y_pred.append(predicted.cpu())
        y_true.append(Y_batch.cpu())
        
    y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)
    loss /= len(loader)
    
    return loss, (y_pred, y_true)

from src.datasets.db6 import DB6MultiSession
#ds = DB6MultiSession(subject=5, sessions=[0], minmax=True, n_classes='7+1', steady=True, image_like_shape=True)
#test_ds = DB6MultiSession(subject=5, sessions=[1], minmax=(ds.X_min, ds.X_max), n_classes='7+1', steady=True, image_like_shape=True)

from sklearn.model_selection import ParameterGrid
patch_size = 20
configs_ = {
    "image_size": (1, 300),
    "patch_size": (1, patch_size),
    "channels": 14,
    "num_classes": 8,

    "dim": [64, 14*20], #14*20, #14 * patch_size,
    #"mlp_dim": 128, #14*20*2, #14 * patch_size * 2,
    "dim_head": [32, 64, 128],
    "heads": [2, 4, 8],
    "depth": [2, 4, 8,],

    "dropout": .2,
    "emb_dropout": 0,

    "pool": "cls",
    "use_cls_token": True,

    "sessions": 5,
}

configs = list(ParameterGrid({k: (v if isinstance(v, list) else [v]) for k, v in configs_.items()}))

configs = sorted(configs, key=lambda x: x["sessions"])

prev_n_sessions = 0
losses_accs_ = []
n_params_ = []
for i, config in enumerate(configs, start=1):
    
    config["mlp_dim"] = 2 * config["dim"]

    n_sessions = config['sessions']

    if n_sessions != prev_n_sessions:
        prev_n_sessions = n_sessions
        ds = DB6MultiSession(folder=os.path.expanduser('~/DB6'), subject=5, sessions=list(range(n_sessions)), minmax=True, n_classes='7+1', steady=True, image_like_shape=True)
        test_ds = DB6MultiSession(folder=os.path.expanduser('~/DB6'), subject=5, sessions=[n_sessions], minmax=(ds.X_min, ds.X_max), n_classes='7+1', steady=True, image_like_shape=True)

    net = ViT(**config)
    
    n_params = sum([param.nelement() for param in net.parameters()])
    n_params_.append(n_params)

    device = 'cuda'

    print(f"Run {i}/{len(configs)}")
    print(config)
    print(f"Params: {n_params / 10**6:.3f}M")
    losses_accs = train(net, ds, k=0, bootstrap="no", test_ds=test_ds, epochs=100)
    losses_accs_.append(losses_accs)
    print()

from pickle import dump

dump([configs, n_params_, losses_accs_], open(f'subject=5,sess=5,epochs=100,lr=1e-3,opt=adam', 'wb'))