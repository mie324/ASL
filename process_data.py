import numpy as np

from sklearn.model_selection import train_test_split

random_seed = 9

names = ['train', 'val', 'test']

def load_data():
    data = np.load('data/image_data.npy')
    labels = np.load('data/image_labels.npy')

    return data, labels

def split_data(data, labels):
    X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels,
                                                                test_size=0.2,
                                                                random_state=random_seed,
                                                                stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.25,
                                                      random_state=random_seed,
                                                      stratify=y_train_val)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_data(datasets):
    for dataset in datasets:
        data, labels = dataset
        for name in names:
            np.save('data/' + name + '_data.npy', data)
            np.save('data/' + name + '_labels.npy', labels)

def counts(datasets):
    for  i, dataset in enumerate(datasets):
        data, labels = dataset
        name = names[i]
        print(name)
        unique, counts = np.unique(labels, return_counts=True)
        value_counts = dict(zip(unique, counts))
        print(value_counts)

def main():
    data, labels = load_data()
    train, val, test = split_data(data, labels)
    save_data((train, val, test))
    counts((train, val, test))

if __name__ == "__main__":
    main()