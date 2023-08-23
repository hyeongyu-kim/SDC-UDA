# SDC-UDA

This is an official pytorch implementation of the  paper 

### [SDC-UDA: Volumetric Unsupervised Domain Adaptation Framework for Slice-Direction Continuous Cross-Modality Medical Image Segmentation (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Shin_SDC-UDA_Volumetric_Unsupervised_Domain_Adaptation_Framework_for_Slice-Direction_Continuous_Cross-Modality_CVPR_2023_paper.pdf) by Hyungseob Shin∗, Hyeongyu Kim∗, Sewon Kim, Yohan Jun, Taejoon Eo and Dosik Hwang.
 

## Notes

Our framework consists of two steps, 1.




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


