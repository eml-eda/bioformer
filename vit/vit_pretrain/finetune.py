import os, sys
from src.datasets.db6 import DB6MultiSession

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

device = 'cuda'
save_model_every_n = 20

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

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_conv = nn.Conv1d(in_channels=channels, out_channels=dim, kernel_size=patch_length, stride=patch_length, padding=0, bias=True)

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
        nn.init.uniform_(self.patch_conv.weight, -bound, bound)
        nn.init.uniform_(self.patch_conv.bias, -bound, bound)
        nn.init.zeros_(self.pos_embedding)
        nn.init.zeros_(self.mlp_head[1].weight)
        nn.init.zeros_(self.mlp_head[1].bias)

    def forward(self, x):
        x = self.patch_conv(x).flatten(2).transpose(-2, -1)

        b, n, _ = x.shape

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

## TRAINING

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

    return losses_accs

from pickle import dump, load

def main_train_subject(subject, training_config):

    n_sessions = 5
    train_sessions = list(range(n_sessions))
    test_sessions = list(range(n_sessions, 10))

    steady=True
    n_classes='7+1'
    bootstrap='no'

    print("Training subject", subject)

    subs = ','.join([str(a) for a in range(1, 11) if a != subject])
    minmax_picklename = f'ds_minmax_sessions={n_sessions}subjects={subs}.pickle'
    print("Loading minmax from", minmax_picklename)
    minmax = load(open(minmax_picklename, 'rb'))

    ds = DB6MultiSession(folder=os.path.expanduser('DB6'), subjects=[subject], sessions=train_sessions, steady=steady, n_classes=n_classes, minmax=minmax).to('cuda')
    test_ds_5 = DB6MultiSession(folder=os.path.expanduser('DB6'), subjects=[subject], sessions=test_sessions, steady=steady, n_classes=n_classes, minmax=(ds.X_min, ds.X_max)).to('cuda')

    test_datasets_steady = [DB6MultiSession(folder=os.path.expanduser('DB6'), subjects=[subject], sessions=[i], 
                                            steady=True, n_classes=n_classes, minmax=(ds.X_min, ds.X_max)) \
                            for i in test_sessions]
    test_datasets = [DB6MultiSession(folder=os.path.expanduser('DB6'), subjects=[subject], sessions=[i], 
                                     steady=False, n_classes=n_classes, minmax=(ds.X_min, ds.X_max)) \
                            for i in test_sessions]
    
    
    results = {}
    results['subject'] = subject
    results['steady'] = steady
    results['n_classes'] = n_classes
    results['bootstrap'] = bootstrap
    results['train_sessions'] = train_sessions
    results['test_sessions'] = test_sessions
    
    for k in [0, 1]:
        result = {}

        net = ViT()
        
        pretrained_path = f"vit_pretrain_subject{subject}_fold{k + 1}.pth"

        net.load_state_dict(torch.load(pretrained_path))
        print("Loaded checkpoint", pretrained_path)

        losses_accs = train(net=net, net_name=f"vit_subject{subject}_fold{k}", ds=ds, k=k, bootstrap=bootstrap, training_config=training_config, test_ds=test_ds_5)
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

    return {'subject': subject}, results

training_config = {
     "epochs": 20,
     "batch_size": 8,
     "optim": "Adam",
     "optim_hparams": {"lr": 1e-4, "betas": (.9, .999), "weight_decay": 0},
     "lr_scheduler": "StepLR",
     "lr_scheduler_hparams": {"gamma": .1, "step_size": 10},
}


main = main_train_subject

from time import time

if __name__ == '__main__':

    all_configs, all_results = [], []
    
    for subject in range(1, 11):
        c, r = main_train_subject(subject=subject, training_config=training_config)
        all_configs.append(c)
        all_results.append(r)
    
    pickle_name = f'results_{time():.0f}.pickle'
    dump([all_configs, all_results], open(pickle_name, 'wb'))
    print("Saved", pickle_name)
