from PIL import Image
import numpy as np
import torch
import os
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns

from load_dataset import *
from util import *
from preprocessing import *

def load_model(path):
    model = torch.load(path, map_location='cpu')
    model = model.eval()
    return model

def load_image(path):
    image = Image.open(path)
    image = image.resize((200,200), Image.ANTIALIAS)
    image = np.asarray(image)
    image = np.transpose(image, (2,0,1))
    image = image/255.0
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    return image

def predict_file(image_path, model_path, config_path):
    model = load_model(model_path)
    config = load_config(config_path)
    try:
        filter = config['filter']
    except KeyError:
        filter = 0
    image = process_image(image_path, filter=filter)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)

    predictions = model(image)
    prediction = predictions.argmax(dim=1).item()
    if prediction < 26:
        letter = chr(prediction + 97)
    elif prediction == 26:
        letter = 'del'
    elif prediction == 27:
        letter = 'nothing'
    elif prediction == 28:
        letter = 'space'

    return letter, prediction, predictions

def predict_folder(folder_path, model_path, config_path):
    images = os.listdir(folder_path)

    labels = np.zeros(len(images))
    predictions = np.zeros(len(images))
    sum = 0.
    for i, image in enumerate(images, 0):
        letter, prediction, discard = predict_file(folder_path+'/' +image, model_path, config_path)
        labels[i] = ord(image[0]) - 97
        if image[0] == letter:
            sum += 1
        predictions[i] = prediction

    accuracy = sum/len(images)

    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix('confusion_matrices/', cm, title='Collected Confusion Matrix')
    print(accuracy)



def predict_test_set(model_path):
    model = load_model(model_path)
    test_loader = load_test_dataset()

    total_err = 0.0
    total_epoch = 0

    actual_labels = []
    predicted_labels = []
    for i, data in enumerate(test_loader, 0):
        input, labels = data
        outputs = model(input)
        prediction = outputs.argmax(dim=1).detach().numpy().tolist()
        labels = labels.long()
        labels = labels.squeeze(1)
        true = labels.detach().numpy().tolist()
        print('Prediction', prediction)
        print('True', true)
        actual_labels += true
        predicted_labels += prediction

        total_err += torch.sum(labels != outputs.argmax(dim=1)).item()
        total_epoch += len(labels)

    err = float(total_err)/total_epoch
    
    #predicted_labels = predicted_labels.argmax(dim=1).detach().numpy()
    #actual_labels = actual_labels.detach().numpy()

    cm = confusion_matrix(actual_labels, predicted_labels)
    plot_confusion_matrix('confusion_matrices/', cm, title='Confusion Matrix')
    sns.heatmap(cm)
    print('Test Error: {:.2f}%'.format(err*100))
    # return predicted_labels, actual_labels
    return cm

def plot_confusion_matrix(path, cm,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.

    Args:
        path: String to the output folderin which to save the cm image
        cm: Numpy array of confusion matrix
        classes: List of strings of names of the classes
        normalize: Whether or not to normalize the cm
        title: Does what you think
        cmap: color map to use
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i','j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'del', 'nothing', 'space']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=270)
    #plt.tight_layout()
    plt.savefig(path + '/' + title)


# if __name__ == "__main__":
#     predict_test_set('models/model_31/model.pt')


