import numpy as np
import scipy.ndimage as ndimg
import scipy.cluster.vq as whitener
import matplotlib.pyplot as plt
from scipy.misc import imsave

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','nothing','space']
alphabet = ['u']
col = 1
row = 1
fig = plt.figure()

for letter, i in zip(alphabet,range(1, col*row + 1)):
    img = ndimg.imread("asl_alphabet_test/"+letter+"_test.jpg")
    fig.add_subplot(row, col, i)
    plt.imshow(img)
plt.show()

fig2 = plt.figure()
for letter, i in zip(alphabet, range(1, col*row + 1)):
    img = ndimg.imread("asl_alphabet_test/"+letter+"_test.jpg")
    img = img[5:-5, 5:-5, :]
    img = whitener.whiten(img)
    img = np.clip(img, 0, 1)
    fig2.add_subplot(row, col, i)
    plt.imshow(img)
plt.show()