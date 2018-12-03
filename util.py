import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
from os import listdir
from shutil import copy
import torch
import pickle
import json
import PIL

def plot(x, valid_acc, train_acc, args, path):
    plt.clf()
    train = 1.0 - savgol_filter(train_acc, 3, 2)
    val = 1.0 - savgol_filter(valid_acc, 3, 2)
    title = 'Training and Validation Accuracy: Batch Size =  ' + str(args['batch_size']) + ' Learn. Rate = ' + str(args['lr'])
    plt.title(title)
    plt.plot(x, train, label='Training')
    plt.plot(x, val, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.legend(loc='best')
    plt.savefig(path + '/accuracy.png')
    # plt.show()

def plot_loss(x, val_loss, train_loss, args, path):
    plt.clf()
    train = savgol_filter(train_loss, 3, 2)
    val = savgol_filter(val_loss, 3, 2)
    title = 'Training and Validation Loss: Batch Size = ' + str(args['batch_size']) + ' Learn. Rate = ' + str(args['lr'])
    plt.title(title)
    plt.plot(x, train, label='Training')
    plt.plot(x, val, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(path+'/loss.png')

def get_path(params):
    file = 'models/model_num.json'
    with open(file, 'r') as fp:
        data = json.load(fp)
    model_num = data['model_num']
    path = 'models/model_%d' % model_num
    os.makedirs(path)
    print('model num%d'% model_num)

    model_num += 1
    data = {'model_num': model_num}
    with open(file, 'w') as fp:
        json.dump(data, fp)

    return path

def load_config(path):
    with open(path) as file:
        config = json.load(file)

    return config

def save_model(model, epoch, val_err, path):
    model_name = '/model.pt'
    torch.save(model, path+model_name)

    data = {'epoch': epoch+1, 'val_err': val_err}
    with open(path+'/info.json', 'w') as fp:
        json.dump(data, fp)

def save_data(path, steps, train_err, train_loss, val_err, val_loss, params):
    data = {'steps': steps,
            'train_err': train_err,
            'train_loss': train_loss,
            'val_err': val_err,
            'val_loss': val_loss,
            'params': params
            }
    batch_size = str(params['batch_size'])
    lr = str(params['lr'])
    epochs = str(params['epochs'])
    data_file = path + '/bs_%s_lr_%s_epochs_%s' % (batch_size, lr, epochs) + '.pkl'
    f = open(data_file, 'wb')
    pickle.dump(data, f)
    f.close()

def copy_files(path):
    copy('model.py', path+'/')
    copy('configuration.json', path+'/')

def display_image(path):
    im = PIL.Image.open(path)
    plt.imshow(im)
    plt.show()

def training_plot(path, file):
    f = open(path + '/' + file, 'rb')
    data = pickle.load(f)
    f.close()
    train_loss = data['train_loss']
    train_acc = data['train_err']
    val_loss = data['val_loss']
    val_acc = data['val_err']
    steps = data['steps']
    params = data['params']
    plot(steps, val_acc, train_acc, params, path)
    plot_loss(steps, val_loss, train_loss, params, path)