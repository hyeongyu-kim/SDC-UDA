import os
import numpy as np
import glob
import medpy.io as medio
import scipy.io as sio
import cv2

#####################################################
################### rename_data #####################
#####################################################


def Rename_predictions(datalist, data_fd, renamed_data_fd):
    
    images_list = [id.strip() for id in open(datalist)]
    epoch = '60000'


    for k in range(len(images_list)):
        print(k, 'k')
        image_orig , image_hdr = medio.load(images_list[k])
        max_val = np.max(image_orig)

        patient_id = images_list[k].split('/')[-1].split('.')[0]
        print(patient_id, 'patient_id')
        traslated_img_dir= sorted(os.listdir(data_fd))
        # filename_all = [ data_fd+'/'+filename+'/'+ filename+'_'+contrast+'.nii.gz'+'\n' for filename in filename_all ]# if filename.endswith('.gz')]
        
        traslated_img_dir = [ filename  for filename in traslated_img_dir if (  epoch in filename and 'to' in filename )]
        id_before_r = 'step' + epoch + '_to_' + str(k) + '.mat'
        
        not_refined_ = sio.loadmat(data_fd +'/' +id_before_r)['fake']
       
       
        assert(not_refined_.shape[-1] == image_orig.shape[-1])
        
        
        # print(not_refined_.shape, 'not_refined__shape+before')
        not_refined_ = cv2.resize(not_refined_ ,(image_orig.shape[0:2]) )
        not_refined_ = ((not_refined_ +1)/2*max_val).astype(np.uint16)
        
        
        
        # print(not_refined_.shape, 'not_refined__shape+after')
        # not_refined_ =( not_refined_ -np.min(not_refined_) )/ (np.max(not_refined_) - np.min(not_refined_)) * 2 - 1 
        medio.save(not_refined_, renamed_data_fd + '/' + patient_id + '_fake_' + epoch + '.nii.gz' , hdr= image_hdr, use_compression = True)
        
        
        # traslated_img = 
        
    # image_path = self.images[item]
    
    
    # if contrast =='t1' and mode =='train':
    
    #     filename_all =sorted(    glob.glob(  os.path.join(data_fd) +'/T1/*.*' ))
    #     filename_all = [filename+ '\n' for filename in filename_all ] 
    #     # print(filename_all, 'filename_all')
        
    # if contrast =='t2' and mode =='train':
    
    #     filename_all =sorted(    glob.glob(  os.path.join(data_fd) +'/*.*' ))
    #     filename_all = [filename+ '\n' for filename in filename_all ] 
    #     # print(filename_all, 'filename_all')
        
    # if contrast =='t2' and mode =='test':
    
    #     filename_all =sorted(    glob.glob(  os.path.join(data_fd) +'/*.*' ))
    #     filename_all = [filename+ '\n' for filename in filename_all ] 
    #     # print(filename_all, 'filename_all')




if __name__ == '__main__':
    
    datalist        = '/home/compu/HG/CrossModa/DRANet-master/DRANet-master/data_list/CrossMODA_3d/t1/train_t1.txt'   
    # data_fd         = '/home/compu/HG/CrossModa/COSMOS_v2/Preds/Timesformer_Deep_seg_go' 
    # data_fd          = '/home/compu/HG/CrossModa/COSMOS_v2/Preds/base_3ch_v2'
    # renamed_data_fd  = '/home/compu/HG/CrossModa/COSMOS_v2/Preds/base_3ch_v2_refined'
    
    data_fd          = '/data/COSMOS_v2_preddir/2D_CNN_NOseg'
    renamed_data_fd  = '/data/COSMOS_v2_preddir/2D_CNN_NOseg_refined'
    
    
    
    # renamed_data_fd = '/home/compu/HG/CrossModa/COSMOS_v2/Preds/Timesformer_Deep_seg_go_psize4_refined'
    os.makedirs(renamed_data_fd , exist_ok=True)
    Rename_predictions(datalist, data_fd, renamed_data_fd)
    print(data_fd,' data_fd')