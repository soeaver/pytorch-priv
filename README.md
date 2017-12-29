# pytorch-priv
Pytorch implementation for Classification, Semantic Segmentation and Object Detection


## Install
* Install [PyTorch>=0.3.0](http://pytorch.org/)
* Clone recursively
  ```
  git clone --recursive https://github.com/soeaver/pytorch-priv
  ```


## Training and Evaluating
1. Modify the `.yml` file in `./cfg/cls/air101-1x64d`
2. Train a network:
     ```
     python train_cls.py --cfg ./cfg/cls/air101-1x64d/air50_1x64d_imagenet.yml 
     ```


## Results

### ImageNet
Single-crop (224x224) validation error rate is reported. 

| Model                       | Flops (M) | Params (M) | Top-1 Error (%) | Top-5 Error (%)  |
| :-------------------------: | --------- |----------- | --------------- | ---------------- |
| AIR50-1x64d                 | 6148.2    | 35.9       | 30.09           | 10.78            |

<div align='center'>
  <img src='data/images/air50_1x64d_curve.png' height='330px'>
</div> 


## Contribute
Feel free to create a pull request if you find any bugs or you want to contribute (e.g., more datasets and more network structures).
