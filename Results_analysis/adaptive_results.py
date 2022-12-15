import os, sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
import numpy as np
from pickle import dump, load
import pickle 
import io 
import matplotlib.pyplot as plt 
sys.path.append('../')
from utils.db6 import DB6MultiSession

folder_results = './artifacts'
folder_data = '../DB6/'
string_old = 'None'
new_test = 0
patient = 3

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

if __name__ == '__main__':
    pickle_name = 'ViT_2_1_32_8_1_14_0_0_results_finetune_0_5_1670954771.pickle'
    results_model1_05 = CPU_Unpickler(open(os.path.join(folder_results,pickle_name), 'rb')).load()
    pickle_name = 'ViT_2_1_32_8_1_14_0_0_results_finetune_5_10_1670559778.pickle'
    results_model1_510 = CPU_Unpickler(open(os.path.join(folder_results,pickle_name), 'rb')).load()
    pickle_name = 'ViT_1_1_8_8_1_14_0_0_results_finetune_0_5_1670554812.pickle'
    results_model2_05 = CPU_Unpickler(open(os.path.join(folder_results,pickle_name), 'rb')).load()
    pickle_name = 'ViT_1_1_8_8_1_14_0_0_results_finetune_5_10_1670955099.pickle'
    results_model2_510 = CPU_Unpickler(open(os.path.join(folder_results,pickle_name), 'rb')).load()
    test_sessions = list(range(5, 10))
    true_overall = 0
    total_overall = 0
    for patient in np.arange(10):
        true = 0
        total = 0
        if patient < 5:
            results_paper = results_model1_05
        else:
            results_paper = results_model1_510
        ds = [DB6MultiSession(folder=os.path.expanduser(folder_data), subjects=[patient+1], sessions=[i], 
                                     steady=True, n_classes='7+1', minmax=True, image_like_shape=True) \
                            for i in test_sessions]
        for sess, test_ds in enumerate(ds):
            test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=False, drop_last=False)
            i = 0
            for X_batch, Y_batch in test_loader:
                predictions = torch.max(results_paper[1][patient]["val-fold"]["outs_steady"][sess][i], axis = 1).indices
                true += sum(predictions==Y_batch)
                total += len(Y_batch)
                true_overall += sum(predictions==Y_batch)
                total_overall += len(Y_batch)
                i+=1
        print(f"Accuracy Subject {patient+1} = {true/total*100}")
    print(f"Accuracy Overall = {true_overall/total_overall*100}")
