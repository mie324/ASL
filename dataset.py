import torch.utils.data as data

class ASLDataset(data.dataset):
    def __init__(self, X, y):
        self.X = x
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]