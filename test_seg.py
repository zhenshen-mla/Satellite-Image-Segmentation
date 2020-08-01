import os
import numpy as np
from dataloader import make_dataloader
from segmentation_model import DeepLab
from metrics import Evaluator
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_pretrained_model(model):
    path = '/home/user/model_params.pkl'
    pretrain_dict = torch.load(path)
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        if k in state_dict:
            print(k)
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    return model

def main():
    nclass = 7
    batch_size = 4
    drop_last = True
    train_loader, val_loader = make_dataloader(batch_size=batch_size, drop_last=drop_last)
    model = DeepLab(num_classes=nclass,
                    output_stride=16,
                    freeze_bn=False)
    model = load_pretrained_model(model)
    evaluator = Evaluator(nclass)
    model = model.cuda()

    model.eval()
    evaluator.reset()

    for i, sample in enumerate(val_loader):
        image, target = sample['image'], sample['label']
        image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)

        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

if __name__ == "__main__":
    main()
