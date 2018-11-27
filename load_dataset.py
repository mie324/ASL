import numpy as np
from dataset import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def load_datasets(batch_size):
    train_data = np.load('data/train_data.npy')
    train_labels = np.load('data/train_labels.npy')
    val_data = np.load('data/val_data.npy')
    val_labels = np.load('data/val_labels.npy')

    train_dataset = ASLDataset(train_data, train_labels)
    val_dataset = ASLDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def load_test_dataset():
    test_data = np.load('data/test_data.npy')
    test_labels = np.load('data/test_labels.npy')

    test_dataset = ASLDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    return test_loader

