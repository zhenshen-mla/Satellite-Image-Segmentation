import numpy as np
import scipy.misc


def rgb_anno(input_path, output_path):
    img = scipy.misc.imread(input_path)
    img = (img >= 128).astype(np.uint8)
    img = 4 * img[:, :, 0] + 2 * img[:, :, 1] + img[:, :, 2]
    myimg = np.zeros((2448, 2448), dtype=np.uint8)
    myimg[img == 3] = 0
    myimg[img == 6] = 1
    myimg[img == 5] = 2
    myimg[img == 2] = 3
    myimg[img == 1] = 4
    myimg[img == 7] = 5
    myimg[img == 0] = 6

    scipy.misc.imsave(output_path, myimg)

if __name__ == '__main__':
    label_dir = '/home/user/label/'
    label2anno_dir = '/home/user/label2anno/'

    for i in range(1, 804):
        input_path = label_dir + str(i) + '.png'
        output_path = label2anno_dir + str(i) + '.png'
        rgb_anno(input_path, output_path)
        print(i)