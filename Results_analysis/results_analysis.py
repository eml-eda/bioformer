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

folder = './artifacts'
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

def print_image4(accuracies, complexities, operations, network_type):
    bar_width = 0.4 
    attention = [True if x == 0 else False for x in network_type ]
    frontend = [True if x == 1 else False for x in network_type ]
    TEMPONet = [True if x == 2 else False for x in network_type ]
    acc_attention = [x for x, y in zip(accuracies, attention) if y == True]
    acc_frontend = [x for x, y in zip(accuracies, frontend) if y == True]
    acc_TEMPONet = [x for x, y in zip(accuracies, TEMPONet) if y == True]

    cpl_attention = [x for x, y in zip(complexities, attention) if y == True]
    cpl_frontend = [x for x, y in zip(complexities, frontend) if y == True]
    cpl_TEMPONet = [x for x, y in zip(complexities, TEMPONet) if y == True]

    opt_attention = [x for x, y in zip(operations, attention) if y == True]
    opt_frontend = [x for x, y in zip(operations, frontend) if y == True]
    opt_TEMPONet = [x for x, y in zip(operations, TEMPONet) if y == True]
    colors = ['#AED6F1', '#21618C', '#E59866', '#D35400', '#7DCEA0', '#196F3D']
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.gcf().subplots_adjust(bottom=0.15,top=0.83)
    plt.grid(axis='y')
    ax1 = plt.scatter(opt_attention, acc_attention, marker = 'o', s = 80, edgecolor = 'k', color=colors[0], label = f'Transformers')
    ax2 = plt.scatter(opt_frontend, acc_frontend, marker = 'o', s = 80, edgecolor = 'k', color=colors[2], label = f'Frontend')
    ax4 = plt.scatter(opt_TEMPONet, acc_TEMPONet, marker = '^', s = 80, edgecolor = 'k', color=colors[5], label = f'TEMPONet')
    # plt.xscale('log')
    plt.legend(fontsize=12,ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    # ax.set_xticks(index)
    # ax.set_xticklabels(['','','1M', '10M'], fontsize=12)#, fontweight='bold')
    plt.xlabel("Complexity [MACs]", fontsize=14, fontweight='bold')
    # ymax = 67
    # ymin = 61
    # ticks = 7
    # plt.locator_params(axis='y', nbins=ticks)
    # n_ticks = int((ymax-ymin)/(ticks-1))
    # plt.ylim(ymin,ymax)
    # labels = []
    # for i in np.arange(ticks):
    #     labels.append(str(ymin + i*n_ticks) + '%')
    # ax.set_yticklabels(labels, fontsize=12)
    plt.ylabel("Accuracy [%]", fontsize=14, fontweight='bold')
    # plt.show()
    plt.savefig("Pareto1.png", dpi=600)
		
    bar_width = 0.4 
    colors = ['#AED6F1', '#21618C', '#E59866', '#D35400', '#7DCEA0', '#196F3D']
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.gcf().subplots_adjust(bottom=0.15,top=0.83)
    plt.grid(axis='y')
    ax1 = plt.scatter(cpl_attention, acc_attention, marker = 'o', s = 80, edgecolor = 'k', color=colors[0], label = f'Transformers')
    ax2 = plt.scatter(cpl_frontend, acc_frontend, marker = 'o', s = 80, edgecolor = 'k', color=colors[2], label = f'Frontend')
    ax4 = plt.scatter(cpl_TEMPONet, acc_TEMPONet, marker = '^', s = 80, edgecolor = 'k', color=colors[5], label = f'TEMPONet')
    # ax2 = plt.plot([1,2,3,4,5], X2_array, '-', marker = '^', markersize = 10, markeredgecolor = 'k', color=colors[2], label = f'Bioformer (h=8,d=1)')
    # ax1 = plt.plot([1,2,3,4,5], X3_array, '--', marker = 'o', markersize = 10, markeredgecolor = 'k', color=colors[1], label = f'Bioformer (h=2,d=2) Pre-Training')
    # ax2 = plt.plot([1,2,3,4,5], X4_array, '--', marker = '^', markersize = 10, markeredgecolor = 'k', color=colors[3], label = f'Bioformer (h=8,d=1) Pre-Training')
    #plt.hlines(np.mean(X1_array),index[0], 12, colors='#AED6F1', linestyles='--')
    #plt.hlines(np.mean(X2_array),index[0], 12, colors='#21618C', linestyles='--')
    # plt.xscale('log')
    plt.legend(fontsize=12,ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    # ax.set_xticks(index)
    # ax.set_xticklabels(['','100k', '150k', '200k','250k', '300k', '350k', '400k', '450k'], fontsize=12)#, fontweight='bold')
    plt.xlabel("Parameters[#]", fontsize=14, fontweight='bold')
    # ymax = 67
    # ymin = 61
    # ticks = 7
    # plt.locator_params(axis='y', nbins=ticks)
    # n_ticks = int((ymax-ymin)/(ticks-1))
    # plt.ylim(ymin,ymax)
    # labels = []
    # for i in np.arange(ticks):
    #     labels.append(str(ymin + i*n_ticks) + '%')
    # ax.set_yticklabels(labels, fontsize=12)
    plt.ylabel("Accuracy [%]", fontsize=14, fontweight='bold')
    plt.savefig("Pareto2.png", dpi=600)

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


if __name__ == '__main__':
    
    complexities = []
    accuracies = []
    operations = []
    network_type = []
    name = []
    for pickle_name in os.listdir(folder):
        if 'result' in pickle_name and 'finetune' in pickle_name:
            results_paper = CPU_Unpickler(open(os.path.join(folder,pickle_name), 'rb')).load()
        else:
            continue
        ###configuration
        try:
            results_paper[0][0]["dim_patch"]
            blocks = 1
            if int(results_paper[0][0]["ch_2"])!= 0:
                blocks +=1
            if int(results_paper[0][0]["ch_3"])!= 0:
                blocks +=1
            tcn_layers = results_paper[0][0]["tcn_layers"]
            patch_size1 = results_paper[0][0]["patch_size1"]
            patch_size2 = results_paper[0][0]["patch_size2"]
            patch_size3 = results_paper[0][0]["patch_size3"]
            dim_patch = results_paper[0][0]["dim_patch"]
            dim_head = results_paper[0][0]["dim_head"]
            heads = results_paper[0][0]["heads"]
            depth = results_paper[0][0]["depth"]
            string = f"Network {pickle_name.split('_')[0]} Layers per Block: {tcn_layers}. Num Blocks: {blocks}. Filters: {patch_size1} {patch_size2} {patch_size3}. Dim_Patch {dim_patch}. Dim_Head {dim_head}. Heads {heads} Depth {depth}"
            if int(pickle_name.split('_')[-3]) == patient and string == string_old:
                continue
            if string != string_old:
                new_test = 1
                string_old = f"Network {pickle_name.split('_')[0]} Layers per Block: {tcn_layers}. Num Blocks: {blocks}. Filters: {patch_size1} {patch_size2} {patch_size3}. Dim_Patch {dim_patch}. Dim_Head {dim_head}. Heads {heads} Depth {depth}"
                print(f"Network {pickle_name.split('_')[0]} Layers per Block: {tcn_layers}. Num Blocks: {blocks}. Filters: {patch_size1} {patch_size2} {patch_size3}. Dim_Patch {dim_patch}. Dim_Head {dim_head}. Heads {heads} Depth {depth}")
            else:
                new_test = 0
            patient = int(pickle_name.split('_')[-3])
        except:
            dim = results_paper[0][0]["dim"]
            print(f"Layers per Block: {1}. Num Blocks: {1}. Filters: (1,10), Dim_Patch {dim}. ")
        ##results
        if new_test == 1:
            acc_overall = 0
        # for sub in np.arange(int(pickle_name.split('_')[-3]), int(pickle_name.split('_')[-2])):
        for sub in np.arange(0,5):
            correct = 0
            total = 0
            for pred, true in zip(results_paper[1][sub]["val-fold"]["y_preds_steady"], results_paper[1][sub]["val-fold"]["y_trues_steady"]):
                correct+= sum(pred==true)
                total+= len(pred)
            acc = correct/total*100
            acc_overall+=acc
            subject = results_paper[1][sub]["subject"]
            print(f"Accuracy of subject {subject}: {acc}")
        if subject == 10:
            print(f"Accuracy Total: {acc_overall/10}\n")
            accuracies.append(acc_overall/10)
            name.append(pickle_name)
            if pickle_name.split('_')[0] == 'temponet':
                
                complexities.append(461512)
                operations.append(11000000)
                network_type.append(2)
            else:
                if blocks == 3:
                    ch = [int(results_paper[0][0]["ch_3"]), int(results_paper[0][0]["ch_2"]), int(results_paper[0][0]["ch_1"])]
                    patch = [int(results_paper[0][0]["patch_size3"][-1]), int(results_paper[0][0]["patch_size2"][-1]), int(results_paper[0][0]["patch_size1"][-1])]
                if blocks == 2:
                    ch = [int(results_paper[0][0]["ch_2"]), int(results_paper[0][0]["ch_1"])]
                    patch = [int(results_paper[0][0]["patch_size2"][-1]), int(results_paper[0][0]["patch_size1"][-1])]
                if blocks == 1:
                    ch = [ int(results_paper[0][0]["ch_1"])]
                    patch = [ int(results_paper[0][0]["patch_size1"][-1])]
                if tcn_layers == 2 or blocks > 1:
                    network_type.append(1)
                else:
                    network_type.append(0)

                cl = parameters_calculator(14, blocks, tcn_layers, dim_patch, dim_head, heads, depth, patch, ch)
                complexities.append(cl)

                ops = ops_calculator(300, 14, blocks, tcn_layers, dim_patch, dim_head, heads, depth, patch, ch)
                operations.append(ops) 
    import pdb;pdb.set_trace()
    print_image4(accuracies, complexities, operations, network_type)
    plt.scatter(complexities,accuracies)
    plt.savefig("prova_cpl.png")
    plt.figure()
    plt.scatter(operations, accuracies)
    plt.savefig("prova_ops.png")
