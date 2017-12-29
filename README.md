# pytorch-priv
Pytorch implementation for Classification, Semantic Segmentation and Object Detection

## Install
* Install [PyTorch>=0.3.0](http://pytorch.org/)
* Clone recursively
  ```
  git clone --recursive https://github.com/soeaver/pytorch-priv
  ```

## Training
1. Modify the `.yml` file in `./cfg/cls/air101-1x64d`
2. Train a network:
     ```
     python train_cls.py --cfg ./cfg/cls/air101-1x64d/air50_1x64d_imagenet.yml 
     ```

## Results

### ImageNet
Single-crop (224x224) validation error rate is reported. 
