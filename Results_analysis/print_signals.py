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
from sklearn.preprocessing import MinMaxScaler
import matplotlib 
import random 

folder_results = './artifacts_patches'
folder_data = '../DB6/'

def print_image(X0, X1, X2, name):
    fig, (ax_left, ax_central, ax_right) = plt.subplots(1,3,figsize=(12, 4))
    plt.gcf().subplots_adjust(bottom=0.15,top=0.83)
    
    ax_left.grid(axis='y')
    ax_right.grid(axis='y')
    ax_central.grid(axis='y')
    ax_left.set_xlabel("Time [samples]", fontsize=14, fontweight='bold')
    ax_left.set_ylabel("Magnitude", fontsize=14, fontweight='bold')
    ax_right.set_ylabel("Magnitude", fontsize=14, fontweight='bold')
    ax_right.set_xlabel("Time [samples]", fontsize=14, fontweight='bold')
    ax_central.set_ylabel("Magnitude", fontsize=14, fontweight='bold')
    ax_central.set_xlabel("Time [samples]", fontsize=14, fontweight='bold')
    fig.subplots_adjust(wspace=.4)
    ax_left.plot(X0)
    ax_central.plot(X1)
    ax_right.plot(X2)
    ax_left.set_ylim(-1,1)
    ax_central.set_ylim(-1,1)
    ax_right.set_ylim(-1,1)
    ax_left.title.set_text('a. Rest Position')
    ax_central.title.set_text('b. Grasp Large Diameter')
    ax_right.title.set_text('c. Adducted Thumb')
    plt.savefig(name, dpi=600)

if __name__ == '__main__':
    patient = 0
    test_sessions = list(range(5, 6))
    scaler = MinMaxScaler()
    ds = [DB6MultiSession(folder=os.path.expanduser(folder_data), subjects=[patient+1], sessions=[i], 
          steady=True, n_classes='7+1', minmax=True, image_like_shape=True) for i in test_sessions]
    random.seed(0)
    for sess, test_ds in enumerate(ds):
        test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=False, drop_last=False)
        i = 0
        for X_batch, Y_batch in test_loader:
            if i == 0:
                X0 = X_batch[0,:5,0,:].T
                X1 = X_batch[-1,:5,0,:].T
                scaler.fit(X1)
            if i == 1:
                X2 = X_batch[-1,:5,0,:].T
                break
            i+=1
        print_image((scaler.transform(X0) - 0.5)*2, (scaler.transform(X1) - 0.5)*2, (scaler.transform(X2) - 0.5)*2, "signals.png")
        print("printed")
        break
