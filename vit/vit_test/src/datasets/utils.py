from torch.utils.data import Dataset

import numpy as np

class SuperSet(Dataset):

    def __init__(self, *datasets):
        self.datasets = datasets
        self.lens = np.cumsum([0] + list(map(len, self.datasets)))

    def __getitem__(self, idx):
        dataset_idx = int(np.argwhere(self.lens > idx)[0]) - 1
        idx = idx - self.lens[dataset_idx]
        return self.datasets[dataset_idx][idx]

    def __len__(self):
        return self.lens[-1]