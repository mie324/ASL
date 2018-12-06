import numpy as np
import scipy.ndimage as ndimg
import scipy.cluster.vq as whitener
import matplotlib.pyplot as plt
from PIL import ImageFilter
from PIL import Image
from PIL import ImageEnhance
from scipy.misc import imsave

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','nothing','space']
#alphabet = alphabet[-10:]
col = 6
row = 5
fig = plt.figure()
brightener = ImageEnhance.Brightness
contraster = ImageEnhance.Contrast
sharpener = ImageEnhance.Sharpness
colourer = ImageEnhance.Color

for letter, i in zip(alphabet,range(1, col*row + 1)):
    img = ndimg.imread("asl_alphabet_test/"+letter+"_test.jpg")
    fig.add_subplot(row, col, i)
    plt.imshow(img)
plt.show()

fig2 = plt.figure()
for letter, i in zip(alphabet, range(1, col*row + 1)):
    img = Image.open("asl_alphabet_test/"+letter+"_test.jpg")
    #img = img[5:-5, 5:-5, :]
    img = brightener(img).enhance(2.7)
    img = contraster(img).enhance(0.5)
    img = sharpener(img).enhance(2.3)
    img = colourer(img).enhance(0.8)
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = np.array(img)
    #img = whitener.whiten(img)
    fig2.add_subplot(row, col, i)
    plt.imshow(img)
    imsave("no_whitten"+letter+".jpg", img)
plt.show()


