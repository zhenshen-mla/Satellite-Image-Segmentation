import os
import matplotlib.pyplot as plt
from PIL import Image

img_dir = '/home/user/image/'
label_dir = '/home/user/label/'
label2anno_dir = '/home/user/label2anno/'

img = Image.open(label2anno_dir + '1.png')
print(img.size)
print(img.format)

plt.figure('Image')
plt.imshow(img)
plt.show()

