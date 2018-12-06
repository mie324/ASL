import numpy as np
import scipy.ndimage as ndimg
import scipy.cluster.vq as whitener
import matplotlib.pyplot as plt
from PIL import ImageFilter
from PIL import Image
from PIL import ImageEnhance
from scipy.misc import imsave

brightener = ImageEnhance.Brightness
contraster = ImageEnhance.Contrast
sharpener = ImageEnhance.Sharpness
colourer = ImageEnhance.Color

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','nothing','space']
#alphabet = alphabet[-10:]
col = 6
row = 5

#just use filters
fig2 = plt.figure()
for letter, i in zip(alphabet, range(1, col*row + 1)):
    img = Image.open("asl_alphabet_test/"+letter+"_test.jpg")
    img = brightener(img).enhance(2.7)
    img = contraster(img).enhance(0.5)
    img = sharpener(img).enhance(2.3)
    img = colourer(img).enhance(0.8)
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = np.array(img)
    fig2.add_subplot(row, col, i)
    plt.imshow(img)
plt.show()

#whiten+enhance
fig2 = plt.figure()
for letter, i in zip(alphabet, range(1, col*row + 1)):
    img = Image.open("asl_alphabet_test/"+letter+"_test.jpg")
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = np.array(img)
    img = whitener.whiten(img)
    img = np.clip(img, 0, 1)
    fig2.add_subplot(row, col, i)
    plt.imshow(img)
plt.show()


#whiten+filters
fig2 = plt.figure()
for letter, i in zip(alphabet, range(1, col*row + 1)):
    img = Image.open("asl_alphabet_test/"+letter+"_test.jpg")
    img = brightener(img).enhance(2.7)
    img = contraster(img).enhance(0.5)
    img = sharpener(img).enhance(2.3)
    img = colourer(img).enhance(0.8)
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = np.array(img)
    img = whitener.whiten(img)
    img = np.clip(img, 0, 1)
    fig2.add_subplot(row, col, i)
    plt.imshow(img)
plt.show()

