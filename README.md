# pytorch-priv
Pytorch implementation for Classification, Semantic Segmentation, Pose Estimation and Object Detection
- [x] **Image Classification**
- [ ] **Semantic Segmentation** (progressing...)
- [ ] **Object Detection** (progressing...)
- [ ] **Pose Estimation** (progressing...)

## Install
* Install [PyTorch>=0.3.0](http://pytorch.org/)
* Install [torchvision>=0.2.0](http://pytorch.org/)
* Clone recursively
  ```
  git clone --recursive https://github.com/soeaver/pytorch-priv
  ```
* pip install easydict


## Training and Evaluating
**For training:**
1. Modify the `.yml` file in `./cfg/cls/air50-1x64d`:
   * the `ckpt` is used to save the checkpoints
   * if you want use cosine learning rate, please set `cosine_lr: True`, then `lr_schedule` and `gamma` will not be used
   * for resuming training, add the `model.pth.tar` to `resume: ` and modify `start_epoch`
   * `rotation`, `pixel_jitter` and `grayscale` are extra data augmentation, recommended for training complex networks only
2. Train a network:
     ```
     python train_cls.py --cfg ./cfg/cls/air101-1x64d/air50_1x64d_imagenet.yml 
     ```

**For evaluating:**
1. Modify the `.yml` file in `./cfg/cls/air50-1x64d`:
   * add the `model.pth.tar` to `pretrained: `
   * set the `evaluate: True`
2. Evaluate a network:
     ```
     python train_cls.py --cfg ./cfg/cls/air101-1x64d/air50_1x64d_imagenet.yml 
     ```


## Features
- [x] [Aligned Inception ResNet (AIR)](https://arxiv.org/abs/1703.06211)
- [x] [Cosine Learning Rate](https://arxiv.org/pdf/1707.06990.pdf) 
- [x] [Mixup](https://arxiv.org/pdf/1710.09412.pdf)


## Results

### ImageNet
Single-crop (224x224) validation error rate is reported. 

| Network                 | Flops (M) | Params (M) | Top-1 Error (%) | Top-5 Error (%) | Speed (im/sec) |
| :---------------------: | --------- |----------- | --------------- | --------------- | -------------- |
| resnet50-1x64d          | 4109.4    | 25.5       | 22.96           | 6.54            | 160.1          |
| air50-1x32d             | 1543.9    | 9.5        | 24.99           | 7.62            | 91.7           |
| air50-1x64d             | 6148.2    | 35.9       | 21.02           | 5.55            | 86.9           |
| air50-1x80d             | 9597.9    | 55.4       | 20.50           | 5.41            | 81.3           |
| air101-1x64d            | 11722.9   | 64.4       | 20.13           | 5.02            | 48.1           |

- Speed test on Single Titan xp with batch-size=1.

<div align='center'>
  <img src='data/images/air50_1x64d_curve.png' height='330px'>
  <img src='data/images/air101_1x64d_curve.png' height='330px'>
</div> 


## License

pytorch-priv is released under the MIT License (refer to the LICENSE file for details).


## Contribute
Feel free to create a pull request if you find any bugs or you want to contribute (e.g., more datasets and more network structures).
