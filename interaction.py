import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from my_deeplab import DeepLab
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import threading


class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return img


class ToTensor(object):
    def __call__(self, img):
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img


class FixScaleCrop(object):
    def __init__(self, crop_size=1000):
        self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        return img


def load_pretrained_model(model):
    p = sys.argv[0].rstrip('segmentation.exe')
    path = p + 'segmentation.pkl'
    pretrain_dict = torch.load(path, map_location=torch.device('cpu'))
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        if k in state_dict:
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    return model


def main(inpath, outpath):
    fix = FixScaleCrop()
    norm = Normalize()
    tot = ToTensor()
    image = Image.open(inpath)
    image = tot(norm(fix(image))).unsqueeze(0)

    model = DeepLab(num_classes=7,
                    output_stride=16,
                    freeze_bn=False)
    model = load_pretrained_model(model)
    # model = model.cuda()
    model.eval()
    # image = image.cuda()
    with torch.no_grad():
        output = model(image)

    pred = output.data.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    pred = np.squeeze(pred)
    plt.imsave(outpath, pred)


class myThread (threading.Thread):   # threading.Thread
    def __init__(self, threadID=1, name='segmentation', inpath='', outpath=''):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.inpath = inpath
        self.outpath = outpath

    def run(self):
        print('Step 2：create new thread：'+self.name, 'predicting, Please wait a moment.')
        main(self.inpath, self.outpath)
        print('Step 3：end this thread：'+self.name, 'run finished.')



class filedialogdemo(QWidget):

    def __init__(self, parent=None):
        super(filedialogdemo, self).__init__(parent)
        # layout = QVBoxLayout()
        layout = QGridLayout()
        self.inpath = ''
        self.count = 0
        self.x = QLabel('Performance of Satellite Image Segmentation Model：mIoU=69.31%, PAcc=87.85%, PAcc_class=81.52%, FWIoU=79.05%')
        self.y = QLabel('Category: Urban land; Agricultural land; Pasture land; Forest; Rivers and lakes; Wasteland; Unknown (cloud, fog)')
        self.a = QLabel('(1) It takes time for CPU to run the CNNs model, so you need to wait about 5~15 seconds after clicking the prediction button.')
        self.b = QLabel('(2) The results save in the root directory of the exe file; please put segmentation.pkl and segmentation.exe in the same path.')
        self.c = QLabel('(3) Do not click the prediction button repeatedly for an input, the result will be overwritten.')
        layout.addWidget(self.x, 0, 0, 1, 5)
        layout.addWidget(self.y, 1, 0, 1, 5)
        layout.addWidget(self.a, 2, 0, 1, 5)
        layout.addWidget(self.b, 3, 0, 1, 5)
        layout.addWidget(self.c, 4, 0, 1, 5)

        self.btn = QPushButton("select input")
        self.btn.clicked.connect(self.loadFile)
        layout.addWidget(self.btn, 5, 0, 2, 1)

        self.btn1 = QPushButton('predict')
        self.btn1.clicked.connect(self.predict)
        layout.addWidget(self.btn1, 5, 1, 2, 5)

        self.label = QLabel()
        self.label.setFixedSize(500, 500)
        self.label.setScaledContents(True)
        layout.addWidget(self.label)

        self.label1 = QLabel()
        self.label1.setFixedSize(500, 500)
        self.label1.setScaledContents(True)
        layout.addWidget(self.label1)


        self.setWindowTitle("Semantic Segmentation Model of Satellite Image")

        self.setLayout(layout)

    def loadFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Image', 'c:\\', 'Image files(*.jpg *.gif *.png)')
        self.inpath = fname
        self.label.setPixmap(QPixmap(fname))
        self.count += 1

    def predict(self):
        p = sys.argv[0].rstrip('segmentation.exe')
        outpath = p + 'result'+str(self.count)+'.png'
        print('Step 1：the inputs path:'+self.inpath, ' the outputs path:'+outpath)
        thread = myThread(inpath=self.inpath, outpath=outpath)
        thread.start()
        thread.join()
        fname = outpath
        self.label1.setPixmap(QPixmap(fname))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    fileload = filedialogdemo()
    fileload.show()
    sys.exit(app.exec_())