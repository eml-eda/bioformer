import os, sys
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
import numpy as np
import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from pickle import dump, load
from time import time
import json 

from utils.db6 import DB6MultiSession
from utils.utils import SuperSet
from utils.model import ViT as ViT
from utils.download_DB6 import download_file
from utils.utils import get_loss_preds
from utils.train import train 
from utils.configs import configs_pretrain, configs_finetune

PROCESSES = 1
save_model_every_n = 20
name_prefix = f"vit_dim64_h8_"

def extend_results(results, result):
    if results is None:
        return result
    for i in range(len(results)): # n campi di results (config, ecc)
        results[i].extend(result[i])
    return results

def main_pretraining(chunk_idx):
    configs = configs_chunks_pretrain[chunk_idx]
    subjects = configs[0]['subjects']
    n_sessions = configs[0]['sessions']
    minmax = True
    if configs[0]["pretrained"] is not None:
        minmax_picklename = f'ds_minmax_sessions={n_sessions}subjects=1,2,3,4,5,6,7,8,9,10.pickle'
        print("Loading minmax from", minmax_picklename)
        minmax = load(open(minmax_picklename, 'rb'))
    ds_config = dict(folder=os.path.expanduser(dataset_folder), subjects=subjects, sessions=list(range(n_sessions)), minmax=minmax, n_classes='7+1', steady=True, image_like_shape=True)
    ds = DB6MultiSession(folder=os.path.expanduser(dataset_folder), subjects=subjects, sessions=list(range(n_sessions)), minmax=minmax, n_classes='7+1', steady=True, image_like_shape=True).to(device)
    add_sub = [a for a in range(1,11) if a not in subjects]
    assert len(add_sub) == 1
    ds_add_sub = DB6MultiSession(folder=os.path.expanduser(dataset_folder), subjects=add_sub, sessions=list(range(n_sessions)), minmax=(ds.X_min, ds.X_max), n_classes='7+1', steady=True, image_like_shape=True).to(device)
    test_ds = DB6MultiSession(folder=os.path.expanduser(dataset_folder), subjects=subjects, sessions=[n_sessions], minmax=(ds.X_min, ds.X_max), n_classes='7+1', steady=True, image_like_shape=True).to(device)
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
        losses_accs = train(net=net, net_name=f"{name_prefix}_{chunk_idx}_{i}", ds=ds, k=0, bootstrap='no', training_config=config['training_config'], test_ds=test_ds, ds_add_sub=ds_add_sub, device = device, save_model_every_n = save_model_every_n)
        losses_accs_.append(losses_accs)
        print()
    return [configs, n_params_, losses_accs_]

def main_finetune(chunk_idx):
    configs = configs_chunks_finetune[chunk_idx]
    subject = configs[0]['subjects']
    n_sessions = configs[0]['sessions']
    train_sessions = list(range(configs[0]['sessions']))
    test_sessions = list(range(configs[0]['sessions'], 10))
    steady=True
    n_classes='7+1'
    bootstrap='no'
    print("Training subject", subject)
    print("Steady", steady)
    print("Bootstrap", bootstrap)
    minmax = True
    if configs[0]["pretrained"] is not None:
        subs = ','.join([str(a) for a in range(1, 11) if a != subject])
        minmax_picklename = f'ds_minmax_sessions={n_sessions}subjects={subs}.pickle'
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
        config['pretrained'] = f"../vit_dim64_h8_{subject - 1}_1_epoch100.pth"
        results = {}
        results['subject'] = subject
        results['steady'] = steady
        results['n_classes'] = n_classes
        results['bootstrap'] = bootstrap
        results['train_sessions'] = train_sessions
        results['test_sessions'] = test_sessions
        result = {}
        net = ViT(**config)
        config['pretrained'] = f"vit_dim64_h8_{9 - (subject - 1)}_1_epoch100.pth"
        if config['pretrained'] is not None:
            net.load_state_dict(torch.load(config['pretrained']))
            print("Loaded checkpoint", config['pretrained'])
        losses_accs = train(net=net, net_name=f"{name_prefix}_{chunk_idx}_{i}", ds=ds, k=k, bootstrap=bootstrap, training_config=config['training_config'], test_ds=test_ds_5, device = device, save_model_every_n = save_model_every_n)
        result['losses_accs'] = losses_accs
        criterion = nn.CrossEntropyLoss()
        test_losses, y_preds, y_trues = [], [], []
        for test_ds in test_datasets_steady:
            test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=False, drop_last=False)
            test_loss, (y_pred, y_true) = get_loss_preds(net, criterion, test_loader, device = device)
            test_losses.append(test_loss)
            y_preds.append(y_pred.cpu())
            y_trues.append(y_true.cpu())
        result['test_losses_steady'] = test_losses
        result['y_preds_steady'] = y_preds
        result['y_trues_steady'] = y_trues

        test_losses, y_preds, y_trues = [], [], []
        for test_ds in test_datasets:
            test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=False, drop_last=False)
            test_loss, (y_pred, y_true) = get_loss_preds(net, criterion, test_loader, device = device)
            test_losses.append(test_loss)
            y_preds.append(y_pred.cpu())
            y_trues.append(y_true.cpu())
        result['test_losses'] = test_losses
        result['y_preds'] = y_preds
        result['y_trues'] = y_trues
        results[f'val-fold'] = result
    results_.append(results)
    return [configs, results_]


configs = list(ParameterGrid({k: (v if isinstance(v, list) else [v]) for k, v in configs_pretrain.items()}))
configs = sorted(configs, key=lambda x: (x["subjects"], x["sessions"]) )
configs_chunks_pretrain = []
dataset_combinations = set(map(lambda x: (x["subjects"], x["sessions"]), configs))
if len(dataset_combinations) == 1:
    for indices in np.array_split(np.arange(len(configs)), PROCESSES):
        if len(indices) > 0: # 
            configs_chunks_pretrain.append(configs[indices[0]:indices[-1]+1])
else:
    prev_dataset_combination, new_chunk = None, None
    for config in configs:
        current_dataset_combination = (config["subjects"], config["sessions"])
        if current_dataset_combination != prev_dataset_combination:
            prev_dataset_combination = current_dataset_combination
            if new_chunk is not None:
                configs_chunks_pretrain.append(new_chunk)
            new_chunk = []
        new_chunk.append(config)
    configs_chunks_pretrain.append(new_chunk)
configs_chunks_idx_pretrain = list(range(len(configs_chunks_pretrain)))


configs = list(ParameterGrid({k: (v if isinstance(v, list) else [v]) for k, v in configs_finetune.items()}))
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

if __name__ == '__main__':
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Missing configuration file: Add config.json file")
        exit(0)
    device = config['device']
    dataset_folder = config['dataset_dir']

    if len(os.listdir(config['dataset_dir'])) == 0:
        for subject in np.arange(1,11):
            for part in ['a', 'b']:
                download_file(subject, part, download_dir = config['dataset_dir'], keep_zip = 'no')
    else:
        print('Dataset already in ./DB6 directory')

    pretrain = True
    finetune = True
    main_pretraining(0)
    if pretrain == True:
        results = None
        with multiprocessing.Pool(PROCESSES) as pool:
            for result in pool.imap_unordered(main_pretraining,  configs_chunks_idx_pretrain):
                results = extend_results(results, result)
        pickle_name = f'results_{time():.0f}.pickle'
        dump(results, open(pickle_name, 'wb'))
        print("Saved", pickle_name)
    
    if finetune == True:
        results = None
        with multiprocessing.Pool(PROCESSES) as pool:
            for result in pool.imap_unordered(main_finetune, configs_chunks_idx_finetune):
                results = extend_results(results, result)
        pickle_name = f'results_{time():.0f}.pickle'
        dump(results, open(pickle_name, 'wb'))
        print("Saved", pickle_name)
