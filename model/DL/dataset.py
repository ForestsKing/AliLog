import numpy as np

from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data, label):
        self.data = data.astype(np.float)
        self.label = label.astype(np.float)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.label[index]
        return X, y

    def __len__(self):
        return len(self.data)