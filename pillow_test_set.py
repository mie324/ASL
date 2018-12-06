import numpy as np
import scipy.ndimage as ndimg
import scipy.cluster.vq as whitener
import matplotlib.pyplot as plt
from PIL import ImageFilter
from PIL import Image
from PIL import ImageEnhance
from scipy.misc import imsave

alphabet = ['a','b','c','d','e','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
alphabet = ['r']
col = 1
row = 1
fig = plt.figure()
brightener = ImageEnhance.Brightness
contraster = ImageEnhance.Contrast
sharpener = ImageEnhance.Sharpness
colourer = ImageEnhance.Color

for letter, i in zip(alphabet,range(1, col*row + 1)):
    img = ndimg.imread("Set2/"+letter+".jpg")
    fig.add_subplot(row, col, i)
    plt.imshow(img)
plt.show()

fig2 = plt.figure()
for letter, i in zip(alphabet, range(1, col*row + 1)):
    img = Image.open("Set2/"+letter+".jpg")
    #img = img[5:-5, 5:-5, :]
    #img = brightener(img).enhance(2.7)
    #img = contraster(img).enhance(1.4)
    #img = sharpener(img).enhance(1.3)
    #img = colourer(img).enhance(0.8)
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = np.array(img)
    img = whitener.whiten(img)
    img = np.clip(img, 0, 1)
    fig2.add_subplot(row, col, i)
    plt.imshow(img)
plt.show()