# BP-Causal
## Overview
This project is the PyTorch implementation for Causal lnference with Sample Balancing for Out-Of-Distribution Detection in Visual Classification [CICAI2022]
## Environment
Python 3.6
Pytorch 1.6.0
matplotlib
yaml
## Dataset
You can get this version of NICO from https://github.com/Wangt-CN/CaaM
## Training
You can train 

  <h1>CUDA_VISIBLE_DEVICES=0 python train.py -cfg conf/ours_resnet18_causal_balance.yaml -gpu -name test</h1>
