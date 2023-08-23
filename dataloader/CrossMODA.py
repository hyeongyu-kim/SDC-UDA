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
import medpy.io as medio

ImageFile.LOAD_TRUNCATED_IMAGES = True

import SimpleITK as sitk




label_colours_CMD = [
    [0, 0, 0],
    [244, 35, 232], 
    [10, 232, 50]]
    
label_colours_CMD = list(map(tuple, label_colours_CMD))

def decode_labels_CMD(imgs, mask, num_images=1, num_classes=3):
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
                    img[int(j_), int(k_)] = label_colours_CMD[int(k)]
                    
        # outputs[i] = np.array(img)
        outputs[i] = img#np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)




class CrossMODA_3d(Cityscapes):
        
    def __init__(self,
                 list_path='./data_list/CrossMODA_3d/',
                 contrast='t1',
                 split='train',
                 crop_size=(256, 256),
                 train=True,
                 numpy_transform=False,
                 in_ch = 1
                 ):
        
        self.list_path = list_path
        self.split = split
        self.crop_size = crop_size
        self.train = train
        self.numpy_transform = numpy_transform
        self.resize = True
        self.contrast= contrast
        self.in_ch = in_ch
        # if self.split =='train' and self.contrast == 't1':
        image_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")
            # label_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")

        if not os.path.exists(image_list_filepath):
            raise Warning("split must be train")

        self.images = [id.strip() for id in open(image_list_filepath)]
        
        self.id_to_trainid = {0: 0, 1: 1, 2: 2}
        
    def __getitem__(self, item):

        image_path = self.images[item]
        
        image , _ = medio.load(image_path)
        image=  torch.tensor(image.astype(np.float))


        ### HERE Activate when change to 2d ####
        # z_choice = random.randint( int((self.in_ch-1)/2) , image.shape[-1]-int((self.in_ch-1)/2)-1)
        # image = image[:,:, z_choice-int((self.in_ch-1)/2) : z_choice+(int(self.in_ch/2)+1)]
        
        
        
        
        
        if len(image.shape)==2 : 
            image=  image[:,:,None]
        
        if self.split =='train' and self.contrast == 't1':
            gt_image_path = image_path.replace('ceT1','Label').replace('T1','Label')
            gt_image, _ =medio.load(gt_image_path)
            gt_image = torch.tensor(gt_image)
            # gt_image = gt_image[:,:, z_choice-int((self.in_ch-1)/2) : z_choice+(int(self.in_ch/2)+1)]

        else : 
            gt_image = torch.zeros_like(image)

        image = image.permute(2,0,1)
        gt_image = gt_image.permute(2,0,1)
        
        
        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.train:
            
            image, gt_image = self._train_sync_transform_BraTS(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform_BraTS(image, gt_image)

        # image=  2*( image - image.min()) / (image.max()-image.min())-1
        return image, gt_image

    def _train_sync_transform_BraTS(self, img, mask):

        if self.resize:

            img = ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.BILINEAR)   ]   )(img)
            if mask is not None: mask=ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.NEAREST)   ]   )(mask)
        
        # img=  torch.cat([img, img, img])
        
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

        
        # img=  torch.cat([img, img, img], dim=0)
        # final transform
        # mask = mask[0]
        
        
        img, mask = self._img_transform_BraTS(img), self._mask_transform_BraTS(mask)
        return img, mask

    def _img_transform_BraTS(self, image):
        
        # image_transforms = ttransforms.Compose([
            # ttransforms.Grayscale(3),
        # ])
        # print(image.size,'image_size-in_img_transform')
        # new_image = image_transforms(image)
        # print(new_image.shape,'image_shape-in_img_transform_after')
        # return new_image
        return image

    def _mask_transform_BraTS(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)

        return target


    def __len__(self):
        return len(self.images)
    
    
    
########################################################################################






class CrossMODA_2d(Cityscapes):
        
    def __init__(self,
                 list_path='./data_list/CrossMODA_2d/',
                 contrast='t1',
                 split='train',
                 crop_size=(256, 256),
                 train=True,
                 numpy_transform=False,
                 in_ch = 1
                 ):
        
        self.list_path = list_path
        self.split = split
        self.crop_size = crop_size
        self.train = train
        self.numpy_transform = numpy_transform
        self.resize = True
        self.contrast= contrast
        self.in_ch = in_ch
        # if self.split =='train' and self.contrast == 't1':
        image_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")
            # label_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")

        if not os.path.exists(image_list_filepath):
            raise Warning("split must be train")

        self.images = [id.strip() for id in open(image_list_filepath)]
        
        self.id_to_trainid = {0: 0, 1: 1, 2: 2}
        
    def __getitem__(self, item):

        image_path = self.images[item]
        
        image , _ = medio.load(image_path)
        image=  torch.tensor(image.astype(np.float))[:,:,None]


        if self.split =='train' and self.contrast == 't1':
            
            gt_image_path = image_path.replace('ceT1','Label').replace('img','label')
            gt_image, _ =medio.load(gt_image_path)
            gt_image = torch.tensor(gt_image)[:,:,None]

        else : 
            gt_image = torch.zeros_like(image)

        image = image.permute(2,0,1)
        gt_image = gt_image.permute(2,0,1)
        
        
        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.train:
            
            image, gt_image = self._train_sync_transform_BraTS(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform_BraTS(image, gt_image)

        # image=  2*( image - image.min()) / (image.max()-image.min())-1
        
        return image, gt_image

    def _train_sync_transform_BraTS(self, img, mask):

        if self.resize:
            img = ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.BILINEAR)   ]   )(img)
            if mask is not None: mask=ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.NEAREST)   ]   )(mask)
        
        # img=  torch.cat([img, img, img])
        
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
        
        # image_transforms = ttransforms.Compose([
            # ttransforms.Grayscale(3),
        # ])
        # print(image.size,'image_size-in_img_transform')
        # new_image = image_transforms(image)
        # print(new_image.shape,'image_shape-in_img_transform_after')
        # return new_image
        return image

    def _mask_transform_BraTS(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)

        return target


    def __len__(self):
        return len(self.images)
    
    
    
########################################################################################




class CrossMODA_25d(Cityscapes):
        
    def __init__(self,
                 list_path='./data_list/CrossMODA_2d/',
                 contrast='t1',
                 split='train',
                 crop_size=(256, 256),
                 train=True,
                 numpy_transform=False,
                 in_ch = 1
                 ):
        
        self.list_path = list_path
        self.split = split
        self.crop_size = crop_size
        self.train = train
        self.numpy_transform = numpy_transform
        self.resize = True
        self.contrast= contrast
        self.in_ch = in_ch
        # if self.split =='train' and self.contrast == 't1':
        image_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")
            # label_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")

        if not os.path.exists(image_list_filepath):
            raise Warning("split must be train")

        self.images = [id.strip() for id in open(image_list_filepath)]
        
        self.id_to_trainid = {0: 0, 1: 1, 2: 2}
        
    def __getitem__(self, item):

        image_path = self.images[item]
        
        
        slice_id = image_path.split('_')[-1].split('.')[0]
        ttt=  '_'+slice_id+'.nii.gz'
        slice_id_previous = '_' + str(int(slice_id)-1)  + '.nii.gz'
        slice_id_after    = '_' + str(int(slice_id)+1)  + '.nii.gz'

        image_path_previous = image_path.replace(ttt, slice_id_previous)
        image_path_after = image_path.replace(ttt, slice_id_after)

        if not os.path.exists(image_path_previous) :
            slice_id_new = str(int(slice_id)+1)
            
        elif not os.path.exists(image_path_after) :
            slice_id_new = str(int(slice_id)-1)
        
        else : 
            slice_id_new = slice_id
            
        ttt=  '_'+slice_id+'.nii.gz'
        slice_id_previous = '_' + str(int(slice_id_new)-1)  + '.nii.gz'
        slice_id_after    = '_' + str(int(slice_id_new)+1)  + '.nii.gz'
        slice_id_current   = '_' + str(int(slice_id_new))  + '.nii.gz'
        image_path_previous = image_path.replace(ttt, slice_id_previous)
        image_path_after = image_path.replace(ttt, slice_id_after)
        image_path = image_path.replace(ttt, slice_id_current)

        
        image_0 , _ = medio.load(image_path_previous)
        image_1 , _ = medio.load(image_path)
        image_2 , _ = medio.load(image_path_after)
        
        image= np.stack([image_0, image_1, image_2], axis=-1).astype(np.float)
        
        image=  torch.tensor(image)

        # print(image.shape, 'iamge_shape')
        # print(image.shape, 'iamge_shape')
        
        # if len(image.shape)==2 : 
        #     image=  image[:,:,None]
        
        
        if self.split =='train' and self.contrast == 't1':
            
            
            gt_image_path = image_path.replace('ceT1','Label').replace('img','label')
            
            
            gt_image_path_previous = gt_image_path.replace(ttt, slice_id_previous)
            gt_image_path_after = gt_image_path.replace(ttt, slice_id_after)
            gt_image_path = gt_image_path.replace(ttt, slice_id_current)

            gt_image0, _ =medio.load(gt_image_path_previous)
            gt_image1, _ =medio.load(gt_image_path)
            gt_image2, _ =medio.load(gt_image_path_after) 

            gt_image= np.stack([gt_image0, gt_image1, gt_image2], axis=-1).astype(np.float)
            gt_image=  torch.tensor(gt_image)
            # gt_image = torch.tensor(gt_image)[:,:,None]
            # gt_image = gt_image[:,:, z_choice-int((self.in_ch-1)/2) : z_choice+(int(self.in_ch/2)+1)]

        else : 
            gt_image = torch.zeros_like(image)

        image = image.permute(2,0,1)
        gt_image = gt_image.permute(2,0,1)
                
        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.train:
            
            image, gt_image = self._train_sync_transform_BraTS(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform_BraTS(image, gt_image)

        # image=  2*( image - image.min()) / (image.max()-image.min())-1
        
        return image, gt_image

    def _train_sync_transform_BraTS(self, img, mask):

        if self.resize:
            img = ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.BILINEAR)   ]   )(img)
            if mask is not None: mask=ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.NEAREST)   ]   )(mask)
        
        # img=  torch.cat([img, img, img])
        
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
        
        # image_transforms = ttransforms.Compose([
            # ttransforms.Grayscale(3),
        # ])
        # print(image.size,'image_size-in_img_transform')
        # new_image = image_transforms(image)
        # print(new_image.shape,'image_shape-in_img_transform_after')
        # return new_image
        return image

    def _mask_transform_BraTS(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)

        return target


    def __len__(self):
        return len(self.images)
    
    




class CrossMODA_25d_Nd(Cityscapes):
        
    def __init__(self,
                 list_path='./data_list/CrossMODA_2d/',
                 contrast='t1',
                 split='train',
                 crop_size=(256, 256),
                 train=True,
                 numpy_transform=False,
                 in_ch = 1
                 ):
        
        self.list_path = list_path
        self.split = split
        self.crop_size = crop_size
        self.train = train
        self.numpy_transform = numpy_transform
        self.resize = True
        self.contrast= contrast
        self.in_ch = in_ch
        # if self.split =='train' and self.contrast == 't1':
        image_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")
            # label_list_filepath = os.path.join(self.list_path, self.contrast, self.split +'_' + self.contrast + ".txt")

        if not os.path.exists(image_list_filepath):
            raise Warning("split must be train")

        self.images = [id.strip() for id in open(image_list_filepath)]
        
        self.id_to_trainid = {0: 0, 1: 1, 2: 2}
        
    def __getitem__(self, item):


        image_path = self.images[item]
        
        half_slice = (self.in_ch-1)//2
        

        slice_id = image_path.split('_')[-1].split('.')[0]
        ttt=  '_'+slice_id+'.nii.gz'
        slice_id_previous = '_' + str(int(slice_id)-int(half_slice))  + '.nii.gz'
        slice_id_after    = '_' + str(int(slice_id)+int(half_slice))  + '.nii.gz'

        image_path_previous = image_path.replace(ttt, slice_id_previous)
        image_path_after = image_path.replace(ttt, slice_id_after)
        

            
        if not os.path.exists(image_path_previous) :
            
            slice_id_new = str(int(slice_id)+int(half_slice))
            
        if not os.path.exists(image_path_after) :
            slice_id_new = str(int(slice_id)-int(half_slice))
        
        else : 
            slice_id_new = slice_id
        
        ttt=  '_'+slice_id+'.nii.gz'
        # slice_id_previous = '_' + str(int(slice_id_new)-half_slice)  + '.nii.gz'
        # slice_id_after    = '_' + str(int(slice_id_new)+half_slice)  + '.nii.gz'
        slice_id_current   = '_' + str(int(slice_id_new))  + '.nii.gz'
        # image_path_previous = image_path.replace(ttt, slice_id_previous)
        # image_path_after = image_path.replace(ttt, slice_id_after)
        image_path = image_path.replace(ttt, slice_id_current)

        image_1 , _ = medio.load(image_path)
        
        
        image  = np.zeros((image_1.shape[0],image_1.shape[1],  self.in_ch))
        image[:,:,half_slice] = image_1
        

        for k in range(1,half_slice+1):



            ttt=  '_'+slice_id_current+'.nii.gz'
            
            slice_id_previous = '_' + str(int(slice_id_new)-k)  + '.nii.gz'
            slice_id_after    = '_' + str(int(slice_id_new)+k)  + '.nii.gz'
            image_path_previous = image_path.replace(slice_id_current, slice_id_previous)
            image_path_after = image_path.replace(slice_id_current, slice_id_after)
            

            d1 , _  =  medio.load(image_path_previous)
            d2 , _  =  medio.load(image_path_after)
            
            image[:,:, half_slice- k] = d1
            image[:,:, half_slice+ k] = d2
        
        
        image=  torch.tensor(image.astype(np.float))

     
     
     
     
        gt_image = torch.zeros_like(image)


        image = image.permute(2,0,1)
        gt_image = gt_image.permute(2,0,1)
                
        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.train:
            
            image, gt_image = self._train_sync_transform_BraTS(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform_BraTS(image, gt_image)

        # image=  2*( image - image.min()) / (image.max()-image.min())-1
        
        return image, gt_image

    def _train_sync_transform_BraTS(self, img, mask):

        if self.resize:
            img = ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.BILINEAR)   ]   )(img)
            if mask is not None: mask=ttransforms.Compose( [ttransforms.Resize( self.crop_size[0] , ttransforms.InterpolationMode.NEAREST)   ]   )(mask)
        
        # img=  torch.cat([img, img, img])
        
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

        
        # img=  torch.cat([img, img, img], dim=0)
        # final transform
        # mask = mask[0]
        
        
        img, mask = self._img_transform_BraTS(img), self._mask_transform_BraTS(mask)
        return img, mask

    def _img_transform_BraTS(self, image):
        
        # image_transforms = ttransforms.Compose([
            # ttransforms.Grayscale(3),
        # ])
        # print(image.size,'image_size-in_img_transform')
        # new_image = image_transforms(image)
        # print(new_image.shape,'image_shape-in_img_transform_after')
        # return new_image
        return image

    def _mask_transform_BraTS(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)

        return target


    def __len__(self):
        return len(self.images)
    
    