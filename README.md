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
1. Modify the `.yml` file in `./cfg/imagenet/air50-1x64d`:
   * the `ckpt` is used to save the checkpoints
   * if you want use cosine learning rate, please set `cosine_lr: True`, then `lr_schedule` and `gamma` will not be used
   * for resuming training, add the `model.pth.tar` to `resume: ` and modify `start_epoch`
   * `rotation`, `pixel_jitter` and `grayscale` are extra data augmentation, recommended for training complex networks only
   
2. Train a network:
     ```
     python cls_train.py --cfg ./cfg/imagenet/air50_1x64d.yml 
     ```
     
    2.1 Training with [mixup](https://arxiv.org/pdf/1710.09412.pdf) (optional):
     ```
     python tools/cls_mixup_train.py --cfg ./cfg/imagenet/air50_1x64d_mixup.yml 
     ```
     for better performace:
     * double the epochs for training with mixup 
     * a few extra epochs with no mixup after the process above

    2.2 Ttraining cifar dataset (optional):
     ```
     python tools/cls_cifar.py --cfg ./cfg/cifar10/resnext29_8x64d.yml
     ```
     or with mixup (usually `weight_decay: 0.0001`):
     ```
     python tools/cls_mixup_cifar.py --cfg ./cfg/cifar10/resnext29_8x64d_mixup.yml
     ```

**For evaluating:**
1. Modify the `.yml` file in `./cfg/imagenet/air50-1x64d`:
   * add the `model.pth.tar` to `pretrained: `
   * set the `evaluate: True`
   
2. Evaluate a network:
     ```
     python train_cls.py --cfg ./cfg/imagenet/air50_1x64d.yml 
     ```
     
**For evaluating image by image:**
1. Modify the `tools/cls_eval.py` file
   
2. Evaluate a network:
     ```
     python tools/cls_eval.py
     ```


## Features
- [x] [Aligned Inception ResNet (air)](https://arxiv.org/abs/1703.06211)
- [x] [Cosine Learning Rate](https://arxiv.org/pdf/1707.06990.pdf) 
- [x] [Mixup](https://arxiv.org/pdf/1710.09412.pdf)
- [x] [Random-Erasing](https://arxiv.org/pdf/1708.04896.pdf)


## Results

### ImageNet1k
Single-crop (224x224) validation error rate is reported. 

| Network                 | Flops (M) | Params (M) | Top-1 Error (%) | Top-5 Error (%) | Speed (im/sec) |
| :---------------------: | --------- |----------- | --------------- | --------------- | -------------- |
| resnet50-1x64d          | 4342.1    | 25.5       | 23.52           | 7.01            | 157.1          |
| resnet101-1x64d         | 8039.0    | 44.5       | 22.18           | 6.23            | 91.7           |


- Speed test on single Titan xp GPU with `batch_size: 1`.


### Cifar10 & Cifar100
Validation error rate is reported. 

| Network                  | Flops (M) | Params (M) | Cifar10 Top-1<br/>Error (%) | Cifar100 Top-1<br/>Error (%) |
| :----------------------: | --------- |----------- | --------------------------- | ---------------------------- |
| resnext29-8x64d          | 5387.2    | 34.4       | 3.73                        | 18.55                        |
| resnext29-8x64d-mixup    | 5387.2    | 34.4       | 2.90                        | --                           |
| resnext29-8x64d-re       | 5387.2    | 34.4       | 3.55                        | --                           |


## License

pytorch-priv is released under the MIT License (refer to the LICENSE file for details).


## Contribute
Feel free to create a pull request if you find any bugs or you want to contribute (e.g., more datasets and more network structures).
