import os
import numpy as np
import glob
import medpy.io as medio
import scipy.io as sio
import cv2

#####################################################
################### rename_data #####################
#####################################################


def Rename_predictions(datalist, data_fd, renamed_data_fd, contrast):
    
    images_list = [id.strip() for id in open(datalist)]
    epoch = '30000'


    for k in range(len(images_list)):
        print(k, 'k')
        image_orig , image_hdr = medio.load(images_list[k])
        max_val = np.max(image_orig)

        patient_id = images_list[k].split('/')[-1].split('.')[0]
        print(patient_id, 'patient_id')
        traslated_img_dir= sorted(os.listdir(data_fd))
        # filename_all = [ data_fd+'/'+filename+'/'+ filename+'_'+contrast+'.nii.gz'+'\n' for filename in filename_all ]# if filename.endswith('.gz')]
        
        traslated_img_dir = [ filename  for filename in traslated_img_dir if (  epoch in filename and contrast in filename )]
        id_before_r = 'step' + epoch +  '_sy_' +  contrast + '_' +str(k) + '.mat'
        
        # print(traslated_img_dir, 'traslated_img_dir')
        # print(id_before_r, 'id_before_r')
        
        
        
        not_refined_ = sio.loadmat(data_fd +'/' +id_before_r)['fake']
        # print(not_refined_.shape,'not_refined_')

       
        assert(not_refined_.shape[-1] == image_orig.shape[-1])
        
        not_refined_ = cv2.resize(not_refined_ ,(image_orig.shape[0:2]) )
        not_refined_ = ((not_refined_ +1)/2*max_val).astype(np.uint16)
        
        medio.save(not_refined_, renamed_data_fd + '/' + patient_id + '_fake_' + epoch + '.nii.gz' , hdr= image_hdr, use_compression = True)
        
        
        



if __name__ == '__main__':
    
    
    # datalist        = '/home/compu/HG/CrossModa/DRANet-master/DRANet-master/data_list/SYNAPSE/T2/train_T2.txt'   
    # contrast        =  't2'
    
    datalist        = '/home/compu/HG/CrossModa/DRANet-master/DRANet-master/data_list/SYNAPSE/T1/train_T1.txt'   
    contrast        =  't1'
    
    
    data_fd          = '/data/COSMOS_v2_preddir/TIME_DEEP_SYNAPSE_T1_CT_V2_Psize4'
    
    renamed_data_fd  = data_fd + '_refined'
    
    
    
    # renamed_data_fd = '/home/compu/HG/CrossModa/COSMOS_v2/Preds/Timesformer_Deep_seg_go_psize4_refined'
    os.makedirs(renamed_data_fd , exist_ok=True)
    Rename_predictions(datalist, data_fd, renamed_data_fd , contrast)
    print(data_fd,' data_fd')
    print(renamed_data_fd, 'refined_data_fd')