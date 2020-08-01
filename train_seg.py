import os
import numpy as np
from tensorboardX import SummaryWriter
import torch
from dataloader import make_dataloader
from segmentation_model import DeepLab
from loss import SegmentationLosses
from lr_scheduler import LR_Scheduler
from metrics import Evaluator


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
writer = SummaryWriter(comment='_vis')

def main():
    nclass = 7
    lr = 0.01
    num_epochs = 400
    batch_size = 4
    best_pred = 0.0
    drop_last = True

    train_loader, val_loader = make_dataloader(batch_size=batch_size, drop_last=drop_last)

    model = DeepLab(num_classes=nclass,
                    output_stride=16,
                    freeze_bn=False)

    train_params = [{'params': model.get_1x_lr_params(), 'lr': lr},
                    {'params': model.get_10x_lr_params(), 'lr': lr * 10}]

    optimizer = torch.optim.SGD(train_params, momentum=0.9, weight_decay=5e-4, nesterov=False)
    criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode='ce')
    evaluator = Evaluator(nclass)
    scheduler = LR_Scheduler(mode='poly', base_lr=lr, num_epochs=num_epochs, iters_per_epoch=len(train_loader))
    model = model.cuda()

    for epoch in range(num_epochs):

        train_loss = 0.0
        count = 0
        model.train()
        for i, sample in enumerate(train_loader):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()
            scheduler(optimizer, i, epoch, best_pred)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            count += 1
        train_loss = train_loss / count
        print('Training  [Epoch: %d, best_pred: %.4f, numImages: %5d, Loss: %.3f]' % (epoch, best_pred, i * batch_size + image.data.shape[0], train_loss))
        writer.add_scalar('scalar/loss_seg_train', train_loss, epoch)
        
        model.eval()
        evaluator.reset()
        test_loss = 0.0
        count = 0
        for i, sample in enumerate(val_loader):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = model(image)
            loss = criterion(output, target)
            test_loss += loss.item()
            count += 1
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(target, pred)

        # Fast test during the training
        test_loss = test_loss / count
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation  [Epoch: %d, numImages: %5d, Loss: %.3f]' % (epoch, i * batch_size + image.data.shape[0], test_loss))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        writer.add_scalar('scalar/loss_seg_val', test_loss, epoch)
        writer.add_scalar('scalar/Acc_val', Acc, epoch)
        writer.add_scalar('scalar/Acc_class_val', Acc_class, epoch)
        writer.add_scalar('scalar/mIou_val', mIoU, epoch)
        writer.add_scalar('scalar/FWIou_val', FWIoU, epoch)
        path = '/home/user/'
        if mIoU > best_pred:
            best_pred = mIoU
            torch.save(model.state_dict(), path+'model_params.pkl')
            print('save the segmentation ' + str(epoch) + 'model, replace the previous parameters')

if __name__ == "__main__":
    main()
    writer.close()
