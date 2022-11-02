import os, sys
import torch
from torch import nn, einsum
from torch.utils.data import DataLoader
import numpy as np

from utils.utils import SuperSet
from utils.utils import get_loss_preds

def train(net, net_name, ds, k, bootstrap, training_config, test_ds=None, ds_add_sub=None, device = 'gpu', save_model_every_n = 1):

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
            train_ds, val_ds = ds, [ds[0]]
            if ds_add_sub != None:
                train_ds_add_sub, _ = ds_add_sub.split(total_folds=2, val_fold=k)
            train_ds = SuperSet(train_ds, train_ds_add_sub)
            if train_loader is None:
                print("Dataset lengths:", len(train_ds), len(val_ds))
            pin_mem = (train_ds[0][0].device.type == 'cpu')
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_mem, drop_last=True)
            train_loader_ = DataLoader(train_ds, batch_size=1024, shuffle=False, pin_memory=pin_mem, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=pin_mem, drop_last=False)
            if test_ds is not None:
                test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=pin_mem, drop_last=False)

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
        train_loss, (y_pred, y_true) = get_loss_preds(net, criterion, train_loader_, device = device)
        train_preds = y_pred.bincount(minlength=(y_true.max() + 1))
        train_acc = (y_true == y_pred).sum() / len(y_true)
        val_loss, (y_pred, y_true) = get_loss_preds(net, criterion, val_loader, device = device)
        val_preds = y_pred.bincount(minlength=(y_true.max() + 1))
        val_acc = (y_true == y_pred).sum() / len(y_true)
        test_loss, test_preds, test_acc = 0, '', 0
        if test_ds is not None:
            test_loss, (y_pred, y_true) = get_loss_preds(net, criterion, test_loader, device = device)
            test_preds = y_pred.bincount(minlength=(y_true.max() + 1))
            test_acc = (y_true == y_pred).sum() / len(y_true)
        with np.printoptions(precision=3, suppress=True):
            print(f"Epoch {epoch + 1:02d}, Train loss {train_loss:.3f}, Train acc {train_acc:.3f} {train_preds:},\n{' '* 20}Val loss {val_loss:.3f} Val acc {val_acc:.3f}, {val_preds},\n{' '* 40}Test loss {test_loss:.3f} Test acc {test_acc:.3f}, {test_preds}")
            losses_accs.append({'train_loss': float(train_loss), 'train_acc': float(train_acc), 'val_loss': float(val_loss), 'val_acc': float(val_acc), 'test_loss': float(test_loss), 'test_acc': float(test_acc)})
        if ((epoch + 1) % save_model_every_n) == 0:
            torch.save(net.state_dict(), f"{net_name}_epoch{epoch+1}.pth")
    return losses_accs
