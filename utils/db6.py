import torch
from torch.utils.data import Dataset, Subset

from scipy.io import loadmat
import numpy as np

from collections.abc import Sequence
from sklearn.utils import resample

import os

from .utils import SuperSet

def windowing(X_instants, R_instants, Y_instants, v_Hz=2000, window_time_s=.150, relative_overlap=.9, steady=True, steady_margin_s=1.5):
    """
        steady=True, steady_margin_s=0 -> finestre dove sample tutti della stessa label (o solo movimento o solo rest)
        steady=True, steady_margin_s=1.5 -> finestre dove tagli i primi e ultimi 1.5s di movimento
        steady=False -> tutte le finestre, anche quelle accavallate tra movimento e rest
    """
    
    # Centro della finestra (numero di campioni)
    #r = round((v_Hz * window_time_s - 1) / 2)
    r = int((v_Hz * window_time_s) / 2)
    # Ampiezza finestra
    #N = 2 * r + 1
    N = 2 * r
    # Campioni fuori finestra da guardare per capire se steady
    margin_samples = round(v_Hz * steady_margin_s)

    overlap_pixels = round(v_Hz * relative_overlap * window_time_s)
    slide = (N - overlap_pixels)
    M_instants, C = X_instants.shape
    # M = Numero di finestre
    M = (M_instants - N) // slide + 1 * int(((M_instants - N) % slide)!=0)
    
    # La label dovrebbe essere quello indicato nell'ultimo istante
    #Y_windows = Y_instants[-1 + N : M_instants : slide]
    Y_windows = Y_instants[r : M_instants - r : slide]
    # La repetition è quello che viene indicato a metà della finestra
    R_windows = R_instants[r : M_instants - r : slide]

    X_windows = np.zeros((M, N, C))
    is_steady_windows = np.zeros(M, dtype=bool)
    for m in range(M):
        c = r + m * slide # c is python-style
        
        #X_windows[m, :, :] = X_instants[c - r : c + r + 1, :]
        X_windows[m, :, :] = X_instants[c - r : c + r, :]

        if Y_instants[c] == 0: # rest position is not margined
            #is_steady_windows[m] = len(set(Y_instants[c - r: c + r + 1])) == 1
            is_steady_windows[m] = len(set(Y_instants[c - r: c + r])) == 1
        else:
            #is_steady_windows[m] = len(set(Y_instants[c - r - margin_samples : c + r + margin_samples + 1])) == 1
            is_steady_windows[m] = len(set(Y_instants[c - r - margin_samples : c + r + margin_samples])) == 1
    
    if steady:
        return X_windows[is_steady_windows], R_windows[is_steady_windows], Y_windows[is_steady_windows]
    return X_windows, R_windows, Y_windows

def read_session(filename):
    annots = loadmat(filename)

    X = annots['emg'][:, np.r_[0:8,10:16]]
    R = annots['rerepetition'].squeeze()
    y = annots['restimulus'].squeeze()

    # Fix class numbering (id -> index)
    y[y >= 3 ] -= 1
    y[y >= (6 - 1)] -= 1
    y[y >= (8 - 2)] -= 1
    y[y >= (9 - 3)] -= 1

    return X, R, y

class DB6Session(Dataset):

    def __init__(self, filename):
        self.X, self.R, self.Y = read_session(filename)
        self.X_min, self.X_max = self.X.min(axis=0), self.X.max(axis=0)
    
    def minmax(self, minmax=None):
        if isinstance(minmax, Sequence) and minmax[0] is not None:
            X_min, X_max = minmax
        else:
            X_min, X_max = self.X_min, self.X_max
        
        X_std = (self.X - X_min) / (X_max - X_min)
        X_scaled = X_std * 2 - 1
        self.X = X_scaled
        return self

    def windowing(self, steady=True, n_classes='7+1', image_like_shape=False, **kwargs):
        if str(n_classes) not in {'7+1', '7'}:
            raise ValueError('Wrong n_classes')

        X_windows, R_windows, Y_windows = windowing(self.X, self.R, self.Y, steady=steady, **kwargs)
        
        if n_classes == '7':
            # Filtra via finestre di non movimento
            mask = Y_windows != 0
            X_windows, R_windows, Y_windows = X_windows[mask], R_windows[mask], Y_windows[mask]
            # Rimappa label da 1-7 a 0-6
            Y_windows -= 1

        self.X = torch.tensor(X_windows, dtype=torch.float32).permute(0, 2, 1)
        if image_like_shape:
            self.X = self.X.unsqueeze(dim=2)
        self.Y = torch.tensor(Y_windows, dtype=torch.long)
        self.R = R_windows

        return self
    
    def to(self, device):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        #self.R = self.R.to(device)
        
        return self

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.Y.shape[0]

    def split(self, total_folds, val_fold=0):
        indices = np.arange(self.R.shape[0])
        train_mask = self.R % total_folds != val_fold
        return Subset(self, indices[train_mask]), Subset(self, indices[~train_mask])

    def split_0(self, total_folds, val_fold=0):
        indices = np.arange(self.R.shape[0])
        train_mask = self.R % total_folds != val_fold
        train_indices = indices[train_mask]
        val_indices = indices[~train_mask]

        train_indices_y = self[train_indices][1]
        train_indices_y0 = train_indices[train_indices_y == 0]
        train_indices_y1 = train_indices[train_indices_y != 0]
        sample_count_per_class_avg = len(train_indices_y1) // 7
        train_indices_y0 = resample(train_indices_y0, n_samples=sample_count_per_class_avg, replace=False)
        train_indices = np.concatenate([train_indices_y0, train_indices_y1], axis=0)

        return Subset(self, train_indices), Subset(self, val_indices)

    def split_1(self, total_folds, val_fold=0):
        indices = np.arange(self.R.shape[0])
        train_mask = self.R % total_folds != val_fold
        train_indices = indices[train_mask]
        val_indices = indices[~train_mask]

        train_indices_y = self[train_indices][1]
        train_indices_y0 = train_indices[train_indices_y == 0]
        train_indices_y1 = train_indices[train_indices_y != 0]
        train_indices_y1 = resample(train_indices_y1, n_samples=len(train_indices_y0) * 7, replace=True)
        train_indices = np.concatenate([train_indices_y0, train_indices_y1], axis=0)

        return Subset(self, train_indices), Subset(self, val_indices)

class DB6MultiSession(SuperSet):

    def __init__(self, subjects, sessions, folder='.', minmax=False, **kwargs):
        self.sessions = [DB6Session(os.path.join(folder, f'S{subject}_D{(i // 2) + 1}_T{(i % 2) + 1}.mat')) for i in sessions for subject in subjects]
        
        if minmax is True: # Apply global minmax
            self.X_min = np.vstack([session.X_min for session in self.sessions]).min(axis=0)
            self.X_max = np.vstack([session.X_max for session in self.sessions]).max(axis=0)
            for session in self.sessions:
                session.minmax(minmax=(self.X_min, self.X_max))
        elif minmax is not False:
            self.X_min, self.X_max = minmax
            for session in self.sessions:
                session.minmax(minmax=minmax)
        
        for session in self.sessions:
            session.windowing(**kwargs)

        # After windowing, each session-dataset length changes, so initialize SuperSet handling here
        super().__init__(*self.sessions)
        
    def to(self, device):
        for session in self.sessions:
            session.to(device)
        return self

    def split(self, total_folds, val_fold=0):
        train_splits, val_splits = [], []
        for train_split, val_split in map(lambda x: x.split(total_folds=total_folds, val_fold=val_fold), self.sessions):
            train_splits.append(train_split)
            val_splits.append(val_split)

        return SuperSet(*train_splits), SuperSet(*val_splits)

    def split_0(self, total_folds, val_fold=0):
        train_splits, val_splits = [], []
        for train_split, val_split in map(lambda x: x.split_0(total_folds=total_folds, val_fold=val_fold), self.sessions):
            train_splits.append(train_split)
            val_splits.append(val_split)

        return SuperSet(*train_splits), SuperSet(*val_splits)

    def split_1(self, total_folds, val_fold=0):
        train_splits, val_splits = [], []
        for train_split, val_split in map(lambda x: x.split_1(total_folds=total_folds, val_fold=val_fold), self.sessions):
            train_splits.append(train_split)
            val_splits.append(val_split)

        return SuperSet(*train_splits), SuperSet(*val_splits)