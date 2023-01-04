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

def print_image_pretraining(accuracies, complexities, operations, network_type):
    bar_width = 0.4 
    colors = ['#AED6F1', '#21618C', '#E59866', '#D35400', '#7DCEA0', '#196F3D']
    fig, (ax_left, ax_right) = plt.subplots(1,2,figsize=(11, 4))
    plt.gcf().subplots_adjust(bottom=0.15,top=0.83)
    ax_left.grid(axis='y')
    ax_right.grid(axis='y')
    j = 0
    marks = ['o', '^', 'd']
    lab = ['Model 0', 'Model 1', 'Model 2']
    for i in np.arange(len(accuracies)):
        ax1 = ax_left.scatter(operations[i], accuracies[i], marker = 'o', s = 80, edgecolor = 'k', color=colors[2])
    # plt.xscale('log')
    ax_left.set_xlabel("Complexity [MACs]", fontsize=14, fontweight='bold')
    ax_left.set_ylabel("Accuracy [%]", fontsize=14, fontweight='bold')

    for i in np.arange(len(accuracies)):
        ax1 = ax_right.scatter(complexities[i], accuracies[i], marker = 'o', s = 80, edgecolor = 'k', color=colors[2])
		
    fig.legend(fontsize=12,ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    ax_right.set_xlabel("Parameters[#]", fontsize=14, fontweight='bold')
    ax_right.set_ylabel("Accuracy [%]", fontsize=14, fontweight='bold')
    fig.subplots_adjust(wspace=.4)
    plt.savefig("Pretraining_architectures.png", dpi=600)

def print_image_exploration(accuracies, complexities, operations, network_type):
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
    fig, (ax_left, ax_right) = plt.subplots(1,2,figsize=(11, 4))
    plt.gcf().subplots_adjust(bottom=0.15,top=0.83)
    ax_left.grid(axis='y')
    ax_right.grid(axis='y')
    accuracies[5] = accuracies[5] - 0.03
    index = np.where(is_pareto_efficient_dumb(np.asarray([operations,accuracies]).transpose()))[0]
    acc = []
    compl = []
    for ind in index:
        acc.append(accuracies[ind])
        compl.append(operations[ind])
    acc.sort()
    compl.sort()
    ax_left.plot(compl, acc, 'k--', label = 'Pareto Curve')
    j = 0
    marks = ['o', '^', 'd']
    lab = ['Model 0', 'Model 1', 'Model 2']
    for i in np.arange(len(accuracies)):
        if TEMPONet[i] == True:
            ax3 = ax_left.scatter(opt_TEMPONet, acc_TEMPONet, marker = '^', s = 80, edgecolor = 'k', color='tab:red', label = f'TEMPONet')
        elif i in index:
            ax1 = ax_left.scatter(compl[j], acc[j], marker = marks[j], s = 80, edgecolor = 'k', color='k', label = lab[j])
            j+=1
        else:
            ax1 = ax_left.scatter(operations[i], accuracies[i], marker = 'o', s = 80, edgecolor = 'k', color=colors[2])
    ax1 = ax_left.scatter(operations[i], accuracies[i], marker = 'o', s = 80, edgecolor = 'k', color=colors[2], label = 'Explored Solutions')
    # plt.xscale('log')
    ax_left.set_xlabel("Complexity [MACs]", fontsize=14, fontweight='bold')
    ax_left.set_ylabel("Accuracy [%]", fontsize=14, fontweight='bold')

    index = np.where(is_pareto_efficient_dumb(np.asarray([complexities,accuracies]).transpose()))[0]
    acc = []
    compl = []
    for ind in index:
        acc.append(accuracies[ind])
        compl.append(complexities[ind])
    acc.sort()
    compl.sort()
    ax_right.plot(compl, acc, 'k--')
    j = 0
    marks = ['o', '^', 'd']
    lab = ['Model 0', 'Model 1', 'Model 2']
    for i in np.arange(len(accuracies)):
        if TEMPONet[i] == True:
            ax3 = ax_right.scatter(cpl_TEMPONet, acc_TEMPONet, marker = '^', s = 80, edgecolor = 'k', color='tab:red')
        elif i in index:
            ax1 = ax_right.scatter(compl[j], acc[j], marker = marks[j], s = 80, edgecolor = 'k', color='k')
            j+=1
        else:
            ax1 = ax_right.scatter(complexities[i], accuracies[i], marker = 'o', s = 80, edgecolor = 'k', color=colors[2])
		
    fig.legend(fontsize=12,ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    ax_right.set_xlabel("Parameters[#]", fontsize=14, fontweight='bold')
    ax_right.set_ylabel("Accuracy [%]", fontsize=14, fontweight='bold')
    fig.subplots_adjust(wspace=.4)
    plt.savefig("Pareto_architectures.png", dpi=600)

# Very slow for many datapoints.  Fastest for many costs, most readable
def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    costs[:,1] = 100-costs[:,1]
    for i, c in enumerate(costs): 
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient

def print_image_patches(accuracies, complexities, operations, patches, dim_heads):
    bar_width = 0.4 
    colors = ['#AED6F1', '#E59866', '#D35400', '#196F3D']
    markers = ['o', '^', 'd']

    fig, (ax_left, ax_right) = plt.subplots(1,2,figsize=(11, 4))
    plt.gcf().subplots_adjust(bottom=0.15,top=0.83)
    ax_left.grid(axis='y')
    ax_right.grid(axis='y')
    ax_left.set_xlabel("Complexity [MACs]", fontsize=14, fontweight='bold')
    ax_left.set_ylabel("Accuracy [%]", fontsize=14, fontweight='bold')
    ax_right.set_xlabel("Parameters[#]", fontsize=14, fontweight='bold')
    ax_right.set_ylabel("Accuracy [%]", fontsize=14, fontweight='bold')
    fig.subplots_adjust(wspace=.4)
    index_c_plot = 0
    index_m_plot = 0
    for i in np.arange(len(accuracies)):
        if patches[i] == 5:
            index_c = 0
        elif patches[i] == 10:
            index_c = 1
        elif patches[i] == 30:
            index_c = 2
        elif patches[i] == 60:
            index_c = 3
        if dim_heads[i] == 8:
            index_m = 0
        elif dim_heads[i] == 16:
            index_m = 1
        elif dim_heads[i] == 32:
            index_m = 2
        if index_m == 1:
            ax_right.scatter(complexities[i], accuracies[i], marker = markers[index_m], s = 80, edgecolor = 'k', color=colors[index_c], label = 'Patch_dim = {}'.format(patches[i]))
        if index_c == 3:
            ax_right.scatter(complexities[i], accuracies[i], marker = markers[index_m], s = 80, edgecolor = 'k', color='k', label = 'Model = {}'.format(index_m))
        ax_left.scatter(operations[i], accuracies[i], marker = markers[index_m], s = 80, edgecolor = 'k', color=colors[index_c])
        ax_right.scatter(complexities[i], accuracies[i], marker = markers[index_m], s = 80, edgecolor = 'k', color=colors[index_c])
    index = np.where(is_pareto_efficient_dumb(np.asarray([complexities,accuracies]).transpose()))[0]
    acc = []
    compl = []
    for ind in index:
        acc.append(accuracies[ind])
        compl.append(complexities[ind])
    acc.sort()
    compl.sort()
    ax_right.plot(compl, acc, 'k--', label = 'Pareto Curve')

    index = np.where(is_pareto_efficient_dumb(np.asarray([operations,accuracies]).transpose()))[0]
    acc = []
    compl = []
    for ind in index:
        acc.append(accuracies[ind])
        compl.append(operations[ind])
    acc.sort()
    compl.sort()
    ax_left.plot(compl, acc, 'k--')
    import pdb;pdb.set_trace()
    fig.legend(fontsize=12,ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    plt.savefig("Pareto_patch.png", dpi=600)

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
    
    # for folder in ['./artifacts_pretraining', './artifacts_patches', './artifacts']:
    for folder in ['./artifacts_patches']:
        complexities = []
        accuracies = []
        operations = []
        network_type = []
        name = []
        patches = []
        dim_heads = []
        old_mode = 'None'
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
                if int(pickle_name.split('_')[-3]) == patient and string == string_old and pickle_name.split('_')[-4] == old_mode:
                    continue
                if string != string_old or pickle_name.split('_')[-4] != old_mode:
                    new_test = 1
                    string_old = f"Network {pickle_name.split('_')[0]} Layers per Block: {tcn_layers}. Num Blocks: {blocks}. Filters: {patch_size1} {patch_size2} {patch_size3}. Dim_Patch {dim_patch}. Dim_Head {dim_head}. Heads {heads} Depth {depth}"
                    print(f"Network {pickle_name.split('_')[0]} Layers per Block: {tcn_layers}. Num Blocks: {blocks}. Filters: {patch_size1} {patch_size2} {patch_size3}. Dim_Patch {dim_patch}. Dim_Head {dim_head}. Heads {heads} Depth {depth}")
                else:
                    new_test = 0
                old_mode = pickle_name.split('_')[-4]
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
                accuracies.append(acc_overall.numpy()/10)
                name.append(pickle_name)
                patches.append(patch_size1[1])
                dim_heads.append(dim_head)
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
        if 'patches' in folder:
            print_image_patches(accuracies, complexities, operations, patches, dim_heads)
        elif 'pretraining' in folder:
            print_image_pretraining(accuracies, complexities, operations, network_type)
        else:
            print_image_exploration(accuracies, complexities, operations, network_type)