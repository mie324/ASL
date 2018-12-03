import torch
import numpy as np
import argparse
from time import time

from model import *
from model_anna import * 
from load_dataset import *
from util import *

def evaluate(model, val_loader, criterion):
    deviceType = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(deviceType)

    model = model.eval()

    total_err = 0.0
    total_loss = 0.0
    total_epoch = 0

    for i, data in enumerate(val_loader, 0):
        instances, labels = data
        labels = labels.long()
        labels = labels.to(device)
        instances = instances.to(device)

        outputs = model(instances)
        labels = labels.squeeze(1)

        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_err += torch.sum(labels != outputs.argmax(dim=1)).item()
        total_epoch += len(labels)

    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i+1)

    model = model.train()

    return err, loss

def train_model(batch_size, lr, epochs, decay, filter, params, path):
    torch.manual_seed(9)
    deviceType = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(deviceType)

    train_loader, val_loader = load_datasets(batch_size, filter)

    model = ASLCNN()
    # model = Net()

    model = model.double()
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_err = np.zeros(epochs)
    train_loss = np.zeros(epochs)
    val_err = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    steps = np.linspace(1, epochs, epochs)

    total_steps = len(train_loader)
    total_train_err = 0.0
    total_train_loss = 0.0
    total_epoch = 0

    best_val_err = 1.0

    start_time = time()

    for epoch in range(epochs):
        total_train_err = 0.0
        total_train_loss = 0.0
        total_epoch = 0

        for i, (instances, labels) in enumerate(train_loader, 0):
            labels = labels.long()
            labels = labels.to(device)
            instances = instances.to(device)

            outputs = model(instances)
            labels = labels.squeeze(1)
            loss = criterion(outputs, labels)

            a = list(model.parameters())[0].clone()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b = list(model.parameters())[0].clone()
            if torch.equal(a.data, b.data):
                print('wtf no change')
                print(list(model.parameters())[0].grad)

            total_train_err += torch.sum(labels != outputs.argmax(dim=1)).item()
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i + 1)
        val_err[epoch], val_loss[epoch] = evaluate(model, val_loader, criterion)

        if val_err[epoch] < best_val_err:
            save_model(model, epoch, val_err[epoch], path)
            best_val_err = val_err[epoch]

        print("Epoch %d: Train err: %0.4f | Train loss: %0.4f | Val err: %0.4f | Val loss: %0.4f" % (epoch + 1,
                                                                                                     train_err[epoch],
                                                                                                     train_loss[epoch],
                                                                                                     val_err[epoch],
                                                                                                     val_loss[epoch]))

    end_time = time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    save_data(path, steps, train_err, train_loss, val_err, val_loss, params)
    plot(steps, val_err, train_err, params, path)
    return model, steps, train_err, train_loss, val_err, val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--decay', type=float, default=1e-04)
    parser.add_argument('--filter', type=int, default=0)
    parser.add_argument('--config', type=str, default='configuration.json')

    args = vars(parser.parse_args())

    if args['config'] != '':
        params = load_config(args['config'])
        print('config')
    else:
        params = args

    batch_size = params['batch_size']
    lr = params['lr']
    epochs = params['epochs']
    decay = params['decay']
    filter = params['filter']

    path = get_path(params)
    copy_files(path)
    model, steps, train_err, train_loss, val_err, val_loss = train_model(batch_size,
                                                                        lr,
                                                                        epochs,
                                                                        decay,
                                                                        filter,
                                                                        params,
                                                                        path)

if __name__ == '__main__':
    main()
