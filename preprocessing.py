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

alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','nothing','space']
#alphabet = alphabet[-10:]
col = 6
row = 5

def process_image(img_path, filter=0):
    img = Image.open(img_path)
    img = img.resize((200,200), Image.ANTIALIAS)

    if filter == 0:
        img = np.array(img)
        img = img/255.0
    elif filter == 1:
        img = filters(img)
        img = img/255.0
    elif filter == 2:
        img = whiten_enhance(img)
    elif filter == 3:
        img = whiten_filters(img)

    img = np.transpose(img, (2, 0, 1))
    return img



# #just use filters
def filters(img):
    img = brightener(img).enhance(2.7)
    img = contraster(img).enhance(0.5)
    img = sharpener(img).enhance(2.3)
    img = colourer(img).enhance(0.8)
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = np.array(img)
    return img

# fig2 = plt.figure()
# for letter, i in zip(alphabet, range(1, col*row + 1)):
#     img = Image.open("asl_alphabet_test/"+letter+"_test.jpg")
#     img = brightener(img).enhance(2.7)
#     img = contraster(img).enhance(0.5)
#     img = sharpener(img).enhance(2.3)
#     img = colourer(img).enhance(0.8)
#     img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
#     img = np.array(img)
#     fig2.add_subplot(row, col, i)
#     plt.imshow(img)
# plt.show()
#
# #whiten+enhance
def whiten_enhance(img):
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = np.array(img)
    img = whitener.whiten(img)
    img = np.clip(img, 0, 1)
    return img

# fig2 = plt.figure()
# for letter, i in zip(alphabet, range(1, col*row + 1)):
#     img = Image.open("asl_alphabet_test/"+letter+"_test.jpg")
#     img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
#     img = np.array(img)
#     img = whitener.whiten(img)
#     img = np.clip(img, 0, 1)
#     fig2.add_subplot(row, col, i)
#     plt.imshow(img)
# plt.show()
#
#
# #whiten+filters
def whiten_filters(img):
    img = brightener(img).enhance(2.7)
    img = contraster(img).enhance(0.5)
    img = sharpener(img).enhance(2.3)
    img = colourer(img).enhance(0.8)
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = np.array(img)
    img = whitener.whiten(img)
    img = np.clip(img, 0, 1)
    return img

# fig2 = plt.figure()
# for letter, i in zip(alphabet, range(1, col*row + 1)):
#     img = Image.open("asl_alphabet_test/"+letter+"_test.jpg")
#     img = brightener(img).enhance(2.7)
#     img = contraster(img).enhance(0.5)
#     img = sharpener(img).enhance(2.3)
#     img = colourer(img).enhance(0.8)
#     img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
#     img = np.array(img)
#     img = whitener.whiten(img)
#     img = np.clip(img, 0, 1)
#     fig2.add_subplot(row, col, i)
#     plt.imshow(img)
# plt.show()
#
