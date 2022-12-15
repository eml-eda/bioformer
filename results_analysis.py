import os, sys
#import torch
#from torch import nn
#from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
import numpy as np
from pickle import dump, load

if __name__ == '__main__':
    #files = ['results_finetune_1669832778.pickle', 'results_finetune_1669961727.pickle',\
    files = ['artifacts/ViT_1_1_32_8_1_14_0_0_results_finetune_0_5_1670496906.pickle','artifacts/ViT_1_1_32_8_1_14_0_0_results_finetune_5_10_1670497061.pickle']
    for pickle_name in files:
        results_paper = load(open(pickle_name, 'rb'))
        ###configuration
        try:
            results_paper[0][0]["dim_patch"]
            blocks = 1
            if results_paper[0][0]["ch_2"]!= None:
                blocks +=1
            if results_paper[0][0]["ch_3"]!= None:
                blocks +=1
            tcn_layers = results_paper[0][0]["tcn_layers"]
            patch_size1 = results_paper[0][0]["patch_size1"]
            patch_size2 = results_paper[0][0]["patch_size2"]
            patch_size3 = results_paper[0][0]["patch_size3"]
            dim_patch = results_paper[0][0]["dim_patch"]
            dim_head = results_paper[0][0]["dim_head"]
            heads = results_paper[0][0]["heads"]
            depth = results_paper[0][0]["depth"]
            print(f"Layers per Block: {tcn_layers}. Num Blocks: {blocks}. Filters: {patch_size1} {patch_size2} {patch_size3}. Dim_Patch {dim_patch}. Dim_Head {dim_head}. Heads {heads} Depth {depth}")
        except:
            dim = results_paper[0][0]["dim"]
            print(f"Layers per Block: {1}. Num Blocks: {1}. Filters: (1,10), Dim_Patch {dim}. ")

        ##results
        acc_overall = 0
        for sub in np.arange(5):
            correct = 0
            total = 0
            for out, pred, true in zip(results_paper[1][sub]["val-fold"]["outs_steady"],results_paper[1][sub]["val-fold"]["y_preds_steady"], results_paper[1][sub]["val-fold"]["y_trues_steady"]):
                correct+= sum(pred==true)
                total+= len(pred)
                print(out)
            acc = correct/total*100
            acc_overall+=acc
            subject = results_paper[1][sub]["subject"]
            print(f"Accuracy of subject {subject}: {acc}")
        print(f"Accuracy Total: {acc_overall/10}\n")

