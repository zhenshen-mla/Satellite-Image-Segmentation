# Satellite-Image-Segmentation

  A prototype system for semantic segmentation of Satellite Images. In the Window's operating system, the size of the executable file is 700MB, model parameters are 40MB, and the computation cost is 266 in GFLOPS.
  
## Introduction
  The system realizes the segmentation of high-resolution satellite images, which includes seven categories, namely Urban land, Agricultural land, Pasture land, Forest, Rivers and lakes, Wasteland, Unknown(cloud, fog). The back-end CNNs model adopts the more advanced DeepLabv3 plus structure. In the model, the prediction is the same size as the input. Therefore, the compression of the input image will affect the prediction accuracy to a certain extent, but considering the model parameters and computation cost, we finally resize to 1000\*1000 from 2448\*2448 before prediction. We package the back-end model into an executable file. And in front-end, we temporarily use QT to design the interface for simple interaction.

## Visualization
![image](https://github.com/zhenshen-mla/Satellite-Image-Segmentation/blob/master/examples/example.jpg)

## Files
  * `/data/train.txt & val.txt`: index of train samples and test samples;
  * `/examples/`: examples of train sample, ground truth and prediction;   
  * `/preprocessing/rgb2anno.py`: three-channel label converted to a single channel;   
  * `/utils/loss.py`: loss function for CNNs model;   
  * `/utils/lr_scheduler.py`: learning rate scheduler;
  * `/utils/metrics.py`: evaluation criteria of semantic segmentation;   
  * `/segmentation_model.py`: DeepLabv3 plus structure;
  * `/train_seg.py & test_seg.py`: model testing and training;
  * `/interaction.py`: the front-end interface;
  
## Requirements  

  Python >= 3.0 
  PyTorch >= 1.0  
  numpy  
  torchvision  
  tensorboardX  
  PyQt5  
  PyInstaller  
  

## Installation
  1. Clone the repo:   
    ```
    git clone https://github.com/zhenshen-mla/Satellite-Image-Segmentation.git   
    ```   
    ```
    cd Satellite-Image-Segmentation   
    ```
  2. For custom dependencies:   
    ```
    pip install tensorboardX matplotlib PyQt5 PyInstaller   
    ```
  
