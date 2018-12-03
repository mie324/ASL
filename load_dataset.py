import numpy as np
from dataset import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from process_data import *

def load_datasets(batch_size, filter):
    # train_data = np.load('data/train_data_' + str(filter) + '.npy')
    # train_labels = np.load('data/train_labels_' + str(filter) + '.npy')
    # val_data = np.load('data/val_data_' + str(filter) + '.npy')
    # val_labels = np.load('data/val_labels_' + str(filter) +'.npy')
    data, labels = load_data(filter)
    train, val, test = split_data(data, labels)

    train_data, train_labels = train
    val_data, val_labels = val

    # np.save('data/test_data_' + str(filter) + '.npy', data)
    # np.save('data/test_labels_' + str(filter) + '.npy', labels)

    train_dataset = ASLDataset(train_data, train_labels)
    val_dataset = ASLDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def load_test_dataset():
    test_data = np.load('data/test_data_0.npy')
    test_labels = np.load('data/test_labels_0.npy')

    test_dataset = ASLDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    return test_loader

