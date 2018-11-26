import numpy as np
import imageio
import torchvision.transforms as transforms
import os

#del = 26, nothing = 27, space = 28
def load_data(path):
    num_samples = 25
    total_imgs = 29*num_samples
    image_data = np.empty((total_imgs, 3, 200, 200), dtype=np.float64)
    labels = np.empty((total_imgs,1), dtype=np.uint8)
    letters = os.listdir(path)
    print(len(letters))
    index = 0
    for letter in letters:
        samples = os.listdir(path+'/'+letter)
        #only going to load 500 for each class for now

        for i, sample in enumerate(samples):
            if i == num_samples:
                break

            image = imageio.imread(path + '/' + letter + '/' + sample)
            image = np.transpose(image, (2, 0, 1))
            image = image/255.0
            image_data[index, ...] = image


            if letter == "del":
                label = 26
            elif letter == "nothing":
                label = 27
            elif letter == "space":
                label = 28
            else:
                label = ord(letter) - 65
            if label > 28:
                print('FUCK: ', letter, label)

            labels[index] = label
            index+= 1
        print(letter, ':', index)

    np.save('data/image_data', image_data)
    np.save('data/image_labels', labels)


if __name__ == '__main__':
    load_data('asl_alphabet_train')
