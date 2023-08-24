# SDC-UDA

This is an official pytorch implementation of the  paper 

### [SDC-UDA: Volumetric Unsupervised Domain Adaptation Framework for Slice-Direction Continuous Cross-Modality Medical Image Segmentation (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Shin_SDC-UDA_Volumetric_Unsupervised_Domain_Adaptation_Framework_for_Slice-Direction_Continuous_Cross-Modality_CVPR_2023_paper.pdf) by Hyungseob Shin∗, Hyeongyu Kim∗, Sewon Kim, Yohan Jun, Taejoon Eo and Dosik Hwang.
 

## Notes
![Figure1_Camera_JPG](https://github.com/hyeongyu-kim/SDC-UDA/assets/77005104/af5da0a7-c72a-41a2-aded-44bb7df45ed9)

Our framework consists of two steps, 1) Image translation and 2) Self-training. In this repository, only step 1 is supported. 
*And we are currently building on it!

## Requirements

(worked on the setting below, and do not guarantee other versions)
```
argparse                1.4.0
numpy                   1.22.3
pytorch                 1.11.0
torchvision             0.12.0
tensorboard             2.9.1
einops                  0.4.1
scikit-learn            1.1.1
scipy                   1.8.1
cudatoolkit             11.3.1
python                  3.9.12
simpleitk               2.0.2
```
(and need to install extra dependencies required.
Information about the packages will be updated!)


## Dataset

You need to modify the data and fit to datalist format provided. We splitted 3D medcial data to every single slice with file, for efficient loading.

The dataset used for training can be downloaded here for each.

CrossMoDA (Vestibular and schwannoma) : https://crossmoda-challenge.ml/

Cardiac (MMWHS) : https://zmiclab.github.io/zxh/0/mmwhs/

## Training

```
python train.py
```

## Logs

You can check the status using 

```
tensorboard --logdir tensorboard/<Your_Experiment_Name>
```

and your checkpoint is saved at /checkpoints.

## Comments

This code was partially borrowed by

https://github.com/Seung-Hun-Lee/DRANet 

https://github.com/lucidrains/segformer-pytorch 

and about the nnU-Net part with Sensitivity & Specificity aware postprocessing is to be updated!
