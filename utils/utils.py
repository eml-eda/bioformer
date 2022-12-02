from torch.utils.data import Dataset
import numpy as np
import torch

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
        
@torch.no_grad()
def get_loss_preds(net, criterion, loader, device = "gpu"):
    y_pred, y_true, out = [], [], []
    loss = 0
    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        outputs = net(X_batch)
        _, predicted = torch.max(outputs, 1)
        loss += criterion(outputs, Y_batch).item()
        y_pred.append(predicted)
        y_true.append(Y_batch)
        out.append(outputs)
    y_pred, y_true = torch.cat(y_pred).cpu(), torch.cat(y_true).cpu()
    loss /= len(loader)
    return loss, (y_pred, y_true, out)