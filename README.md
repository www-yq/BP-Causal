# BP-Causal
## Overview
This project is the PyTorch implementation for Causal lnference with Sample Balancing for Out-Of-Distribution Detection in Visual Classification [CICAI2022]
## Environment
Python 3.6  
Pytorch 1.6.0  
Or you can install directly from the environment.yaml we provide.

    conda env create -f environment.yaml
## Dataset
You can get this version of NICO dataset from [CaaM](https://github.com/Wangt-CN/CaaM)
## Training
Train the model.  

    CUDA_VISIBLE_DEVICES=0 python train.py -cfg conf/ours_resnet18_causal_balance.yaml -gpu -name ours_resnet18_gb
## Evaluation
We put the pretrained model of this method under the checkpoint folder.  

    CUDA_VISIBLE_DEVICES=0 python train.py -cfg conf/ours_resnet18_causal_balance.yaml -debug -gpu -eval checkpoint/resnet18_ours_cbam_multi-144-best.pth
## BiTex
If you find our codes helpful, please cite our paper:

        @inproceedings{wang2022causal,
          title={Causal Inference with Sample Balancing for Out-of-Distribution Detection in Visual Classification},
          author={Wang, Yuqing and Li, Xiangxian and Ma, Haokai and Qi, Zhuang and Meng, Xiangxu and Meng, Lei},
          booktitle={Artificial Intelligence: Second CAAI International Conference, CICAI 2022, Beijing, China, August 27--28, 2022, Revised Selected Papers, Part I},
          pages={572--583},
          year={2022},
          organization={Springer}
        }
## Acknowledgement
Thanks to its authors of [CaaM](https://github.com/Wangt-CN/CaaM) and the NICO and NICO++ dataset.
