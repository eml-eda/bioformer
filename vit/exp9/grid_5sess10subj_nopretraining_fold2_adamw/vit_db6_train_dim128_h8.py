# -*- coding: utf-8 -*-
"""vit_db6_train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UyXezp2XDZssqvahtAPjQ7USiTHp1RGI
"""

# Commented out IPython magic to ensure Python compatibility.
# SETUP

#! [ -f ~/.ssh/id_rsa_github ] || (echo "Downloading repo private key" && (mkdir -p ~/.ssh ; wget -qq https://www.dropbox.com/s/9vw58w194ozr3ov/id_rsa_master-thesis?dl=1 -O ~/.ssh/id_rsa_github ; chmod 400 ~/.ssh/id_rsa_github))
#! (grep --quiet --no-messages github ~/.ssh/config) || (echo "Adding GitHub public key" && echo "github.com ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAq2A7hRGmdnm9tUDbO9IDSwBK6TbQa+PXYPCPy6rbTrTtw7PHkccKrpp0yVhp5HdEIcKr6pLlVDBfOLX9QUsyCOV0wzfjIJNlGEYsdlLJizHhbn2mUjvSAHQqZETYP81eFzLQNnPHt4EVVUh7VfDESU84KezmD5QlWpXLmvU31/yMf+Se8xhHTvKSCZIFImWwoG6mbUoWf9nzpIoaSjB+weqqUUmpaaasXVal72J+UX2B+2RPW3RcT0eOzQgqlJL3RKrTJvdsjE3JEAvGq3lGHSZXy28G3skua2SmVi/w4yCE6gbODqnTWlg7+wC604ydGXA8VJiS5ap43JXiUFFAaQ==" >> ~/.ssh/known_hosts)
#! (grep --quiet --no-messages github ~/.ssh/config) || (echo "Adding repo private key" && echo "IdentityFile ~/.ssh/id_rsa_github" >> ~/.ssh/config)
#! [ -d ~/master-thesis ] || (echo "Cloning codebase" && cd ~ && git clone --quiet git@github.com:francibm97/master-thesis.git)
#! (cd ~/master-thesis && git pull)
#! pip install scipy sklearn pandas
#! pip install einops

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

# DATASET DOWNLOAD

#! mkdir -p ~/DB6
#! cd ~/DB6 && wget -q --show-progress -r --no-parent --no-host-directories --cut-dirs=10 --accept 'S5_*' http://admin:nimda@lino1.francesco.pw/db6/

#! cd ~/DB6 && wget -q --show-progress -r --no-parent --no-host-directories --cut-dirs=10 --accept 'S*_D1_T1.mat' http://admin:nimda@lino1.francesco.pw/db6/
#! cd ~/DB6 && wget -q --show-progress -r --no-parent --no-host-directories --cut-dirs=10 --accept 'S*_D1_T2.mat' http://admin:nimda@lino1.francesco.pw/db6/

PROCESSES = 5
device = 'cuda'
save_model_every_n = 20

from src.datasets.db6 import DB6MultiSession

# MODEL

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
        #return self.norm(self.fn(x, **kwargs))

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
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3,
                 dim_head = 64, dropout = 0., emb_dropout = 0., use_cls_token=True, 
                 sessions="ignore", subjects="ignore", training_config="ignore", pretrained="ignore", chunk_idx="ignore", chunk_i="ignore"):
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
        config["mlp_dim"] = config["dim"] * 2

        net = ViT(**config)
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

            net = ViT(**config)

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
name_prefix = f"vit_dim64_h8"

patch_size = 10
configs_ = {
    "image_size": (1, 300),
    "patch_size": (1, patch_size),
    "channels": 14,
    "num_classes": 8,

    "dim": 64,
    "mlp_dim": 128,
    "dim_head": 32,
    "heads": 8,
    "depth": 1,

    "dropout": .2,
    "emb_dropout": 0,

    "pool": "cls",
    "use_cls_token": True,

    "sessions": 5,
    "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    "pretrained": None,

    "training_config": [
        {
         "epochs": 200,
         "batch_size": 64,
         "optim": "AdamW",
         "optim_hparams": {"lr": 0, "betas": (.9, .999), "weight_decay": 0.01},
         "lr_scheduler": "CyclicLR",
         "lr_scheduler_hparams": {"base_lr": 1e-7, "max_lr": 1e-3, "step_size_up": 150, "step_size_down": None, "mode": 'triangular', "cycle_momentum": False},
        },
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

import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from pickle import dump
from time import time

def is_notebook():
    return 'get_ipython' in globals()

def extend_results(results, result):
    if results is None:
        return result
    
    for i in range(len(results)): # n campi di results (config, ecc)
        results[i].extend(result[i])
    
    return results

#main = main_1test
main = main_train_subject

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

#! python vit_db6_train.py
