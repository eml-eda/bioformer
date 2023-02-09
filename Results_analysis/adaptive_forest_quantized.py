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

folder_results = './artifacts_quantized'
folder_tree = './artifacts_forest'
folder_data = '../DB6/'
string_old = 'None'
new_test = 0
patient = 3

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def multiplyList(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result

def ops_calculator(image_dimension, ch_initial, blocks, tcn_layers, dim_patch, dim_head, heads, depth, patch, ch):
    ops = 0
    for bl in np.arange(blocks):
        if tcn_layers == 2:
            ops += 3*ch_initial*ch[bl]*image_dimension
        if bl != (blocks-1):
            image_dimension = int(image_dimension/patch[bl])
            ops += patch[bl]*ch[bl]*ch[bl]*image_dimension
        else:
            image_dimension = int(image_dimension/patch[bl])
            ops += patch[bl]*dim_patch*ch[bl]*image_dimension
        ch_initial = ch[-blocks+bl]
    ops += depth*(dim_head*heads*dim_patch*image_dimension + dim_patch*dim_patch*2*image_dimension*2 + dim_head*heads*2*image_dimension*image_dimension + dim_patch*dim_head*heads*3*image_dimension)
    return ops

def parameters_calculator(ch_initial, blocks, tcn_layers, dim_patch, dim_head, heads, depth, patch, ch):
    #compute parameters
    cl = 0
    for bl in np.arange(blocks):
        if tcn_layers == 2:
            cl += 3*ch_initial*ch[bl]
        if bl != (blocks-1):
            cl += patch[bl]*ch[bl]*ch[bl]
        else:
            cl += patch[bl]*ch[bl]*dim_patch
        ch_initial = ch[-blocks+bl]
    cl+=64*8
    cl+=300/multiplyList(patch)*64
    cl+=depth*(dim_patch*dim_patch*4+dim_patch*3+dim_patch*dim_head*heads*4)
    return cl

def adaptive_predict(threshold, sm, probs_little, probs_big):
    n_samples, classes = probs_little.shape
    big_used = np.zeros_like(sm)
    mask_changes = sm <= threshold
    big_used[sm <= threshold] = 1
    predictions = np.argmax(probs_little, axis=-1)
    if np.any(mask_changes):
        predictions[sm <= threshold] = np.argmax(probs_big[sm <= threshold], axis=-1)
    return predictions, big_used.mean()

def score_margin(data):
    partial_sort_idx = np.argpartition(-data, kth=1, axis=1)
    partial_sort_val = np.take_along_axis(data, partial_sort_idx, axis=1)[:, :2] 
    sm = np.abs(np.diff(partial_sort_val, axis=1)).reshape(-1)
    return sm

def print_image(accuracies, complexities, static_acc, static_ops):
    bar_width = 0.4 
    colors = ['#AED6F1', '#E59866', '#D35400', '#196F3D']
    markers = ['o', '^', 'd']

    fig, ax_right = plt.subplots(figsize=(6, 4))
    plt.gcf().subplots_adjust(bottom=0.15,top=0.83)
    ax_right.grid(axis='y')
    ax_right.set_xlabel("Operations[#]", fontsize=14, fontweight='bold')
    ax_right.set_ylabel("Accuracy [%]", fontsize=14, fontweight='bold')
    fig.subplots_adjust(wspace=.4)
    # ax_left.scatter(operations[i], accuracies, marker = markers[index_m], s = 80, edgecolor = 'k', color=colors[index_c])
    ax_right.scatter(complexities[:-2], accuracies[:-2], marker = markers[0], s = 80, edgecolor = 'k', color=colors[0], label = 'Adaptive Models')
    # ax_right.scatter(complexities[-2:], accuracies[-2:], marker = markers[0], s = 80, edgecolor = 'k', color=colors[1])
    ax_right.scatter(static_ops, static_acc, marker = markers[0], s = 80, edgecolor = 'k', color=colors[2], label = 'Static Models')
    fig.legend(fontsize=12,ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    plt.savefig("Adaptive.png", dpi=600)

if __name__ == '__main__':
    # pickle_name = 'ViT_2_1_32_8_1_14_0_0_results_finetune_nopretraining_0_10_1673556999.pickle'
    pickle_name = 'ViT_1_1_8_8_1_14_0_0_results_finetune_nopretraining_0_10_1673565397.pickle'
    results_model1_05 = CPU_Unpickler(open(os.path.join(folder_results,pickle_name), 'rb')).load()
    pickle_name = 'ViT_1_1_8_8_1_14_0_0_results_finetune_nopretraining_0_10_1673565397.pickle'
    results_model2_05 = CPU_Unpickler(open(os.path.join(folder_results,pickle_name), 'rb')).load()
    # model1_cl = parameters_calculator(14, 1, 2, 64, 32, 8, 1, [10], [14])
    # model1_ops = ops_calculator(300, 14, 1, 2, 64, 32, 8, 1, [10], [14])
    model1_cl = parameters_calculator(14, 1, 2, 64, 8, 8, 1, [10], [14])
    model1_ops = ops_calculator(300, 14, 1, 2, 64, 8, 8, 1, [10], [14])
    model2_cl = parameters_calculator(14, 1, 1, 64, 8, 8, 1, [10], [14])
    model2_ops = ops_calculator(300, 14, 1, 1, 64, 8, 8, 1, [10], [14])
    test_sessions = list(range(5, 10))
    acc = []
    complexity = []
    true_overall = 0
    total_overall = 0
    big_used_average = 0
    acc_point = 0
    compl_point = 0
    opt_point = 0
    import json
    for patient in np.arange(10):
        true = 0
        total = 0
        if patient < 5:
            results_model1 = results_model1_05
            results_model2 = json.load(open(f"artifacts_forest/s{patient+1}_pred.json"))
        else:
            results_model1 = results_model1_05
            results_model2 = json.load(open(f"artifacts_forest/s{patient+1}_pred.json"))
            
        for sess in np.arange(5):
            if sess == 0:
                model1_results = results_model1[1][patient]["val-fold"]["y_preds_steady"][sess].numpy()
                labels = results_model1[1][patient]["val-fold"]["y_trues_steady"][sess].numpy()
            else:
                model1_results = np.concatenate((model1_results,(results_model1[1][patient]["val-fold"]["y_preds_steady"][sess].numpy())),axis=0)
                labels = np.concatenate((labels,(results_model1[1][patient]["val-fold"]["y_trues_steady"][sess].numpy())),axis=0)
        model2_results = results_model2['test predictions']
        predictions = []
        for i, pred in enumerate(model2_results):
            if pred == 0:
                predictions.append(0)
            else:
                predictions.append(model1_results[i])
        true += sum(predictions == labels)
        total += len(labels)
        true_overall += sum(predictions==labels)
        total_overall += len(labels)
        print(f"Accuracy Subject {patient+1} = {true/total*100} Big Used: {(len(model2_results)-sum(model2_results))/(len(model2_results))}")
        big_used_average += (len(model2_results)-sum(model2_results))/(len(model2_results))
    print(f"Accuracy Overall = {true_overall/total_overall*100} ")
    acc.append(true_overall/total_overall*100)
    print(big_used_average/10*model1_ops)
    import pdb;pdb.set_trace()