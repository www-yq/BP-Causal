# BP-Causal
## Overview
This project is the PyTorch implementation for Causal lnference with Sample Balancing for Out-Of-Distribution Detection in Visual Classification [CICAI2022]
## Environment
Python 3.6  
Pytorch 1.6.0  
matplotlib  
yaml
## Dataset
You can get this version of NICO dataset from [CaaM](https://github.com/Wangt-CN/CaaM)
## Training
train the model
CUDA_VISIBLE_DEVICES=0 python train.py -cfg conf/ours_resnet18_causal_balance.yaml -gpu -name ours_resnet18_gb
## Evaluation
We put the pretrained model of this method under the checkpoint folder.
CUDA_VISIBLE_DEVICES=0 python train.py -cfg conf/ours_resnet18_causal_balance.yaml -debug -gpu -eval checkpoint/resnet18_ours_cbam_multi-144-best.pth
## Acknowledgement
Thanks to its authors of [CaaM](https://github.com/Wangt-CN/CaaM) and the NICO and NICO++ dataset.
