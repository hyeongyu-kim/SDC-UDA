# -*- coding: utf-8 -*-
import random
import scipy.io
from PIL import Image, ImageOps, ImageFilter, ImageFile
import numpy as np
import copy
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms
import nibabel as nib
from .Cityscapes import Cityscapes

ImageFile.LOAD_TRUNCATED_IMAGES = True

import SimpleITK as sitk




label_colours_BraTS = [
    [0, 0, 0],
    [244, 35, 232]]
    
label_colours_BraTS = list(map(tuple, label_colours_BraTS))

def decode_labels_BraTS(imgs, mask, num_images=1, num_classes=2):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    if n < num_images:
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        # img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        
        
        img = (imgs[i].permute(1,2,0).cpu().numpy()+1)/2*255
        
        # pixels = img.load()
        
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes and k> 0:
                    img[int(j_), int(k_)] = label_colours_BraTS[int(k)]
                    
        # outputs[i] = np.array(img)
        outputs[i] = img#np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)




class BraTS2018_2d(Cityscapes):
        
    def __init__(self,
                 list_path='/home/compu/HG/CrossModa/DRANet-master/DRANet-master/data_list/BraTS2018_2d/',
                 contrast='t1',
                 split='train',
                 crop_size=(256, 256),
                 train=True,
                 numpy_transform=False
                 ):
        
        self.list_path = list_path
        self.split = split
        self.crop_size = crop_size
        self.train = train
        self.numpy_transform = numpy_transform
        self.resize = True
        self.contrast= contrast

        image_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")
        label_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")
        
        if not os.path.exists(image_list_filepath):
            raise Warning("split must be train")

        self.images = [id.strip() for id in open(image_list_filepath)]
        self.labels = [id.strip() for id in open(label_list_filepath)]

        self.id_to_trainid = {0: 0, 1: 1, 2: 1, 3: 1, 4:1}
        
    def __getitem__(self, item):

        image_path = self.images[item]
        image = torch.tensor(np.load(image_path))

        gt_image_path = image_path.replace(self.contrast,'seg')
        gt_image = torch.tensor(np.load(gt_image_path))
        
        
        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.train:
            
        
            image, gt_image = self._train_sync_transform_BraTS(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform_BraTS(image, gt_image)

        # image=  2*( image - image.min()) / (image.max()-image.min())-1
        return image, gt_image

    def _train_sync_transform_BraTS(self, img, mask):
            '''
            '''
            if self.resize:
                img = ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.BILINEAR)   ]   )(img)
                # img = img.resize(self.crop_size, Image.BICUBIC)
                if mask is not None: mask=ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.NEAREST)   ]   )(mask)
            
            img=  torch.cat([img, img, img])
            
            # final transform
            if mask is not None:
                mask = mask[0]
                img, mask = self._img_transform_BraTS(img), self._mask_transform_BraTS(mask)
            
                return img, mask
            else:
                img = self._img_transform_BraTS(img)
                return img

    def _val_sync_transform_BraTS(self, img, mask):
        if self.resize:
            img = ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.BILINEAR)   ]   )(img)
            if mask is not None: mask=ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.NEAREST)   ]   )(mask)

        
        img=  torch.cat([img, img, img], dim=0)
        # final transform
        # mask = mask[0]
        
        
        img, mask = self._img_transform_BraTS(img), self._mask_transform_BraTS(mask)
        return img, mask

    def _img_transform_BraTS(self, image):
        
        image_transforms = ttransforms.Compose([
            ttransforms.Grayscale(3),
        ])
        # print(image.size,'image_size-in_img_transform')
        new_image = image_transforms(image)
        # print(new_image.shape,'image_shape-in_img_transform_after')
        return new_image

    def _mask_transform_BraTS(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)

        return target


    def __len__(self):
        return len(self.images)
    
    
    
########################################################################################


