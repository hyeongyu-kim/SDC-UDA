import torch
import torch.utils.data
from dataloader import *



def get_dataset_with_eval(dataset, batch, imsize, workers, args):


    if dataset == 'to':
        
        train_dataset = CrossMODA_25d(list_path='./data_list/CrossMODA_2d', 
                                contrast='t1', split='train',crop_size=imsize, train=True, in_ch = args.in_ch)
        
        test_dataset  = CrossMODA_25d_Nd(list_path='./data_list/CrossMODA_2d', 
                                  contrast='t1', split='train',crop_size=imsize, train=False, in_ch = 14)
        
        eval_dataset = CrossMODA_3d(list_path='./data_list/CrossMODA_3d', 
                                  contrast='t1', split='train',crop_size=imsize, train=False, in_ch = 1)
        
        
    elif dataset == 'tt':
        
        train_dataset = CrossMODA_25d(list_path='./data_list/CrossMODA_2d', 
                                    contrast='t2', split='train',crop_size=imsize, train=True, in_ch = args.in_ch)
        
        test_dataset  = CrossMODA_25d_Nd(list_path='./data_list/CrossMODA_2d', 
                                  contrast='t2', split='train',crop_size=imsize, train=False, in_ch = 14)

        eval_dataset = CrossMODA_3d(list_path='./data_list/CrossMODA_3d', 
                                  contrast='t2', split='train',crop_size=imsize, train=False, in_ch = 1)




        
    elif dataset == 'mr_fudan':
        
        train_dataset = FUDAN_25d_Nd(list_path='./data_list/FUDAN_2d', 
                                      contrast='mr', split='train',crop_size=imsize, train=True, in_ch = args.in_ch)
            
        test_dataset  = FUDAN_25d(list_path='./data_list/FUDAN_2d', 
                                  contrast='mr', split='train',crop_size=imsize, train=False, in_ch = 1)

        eval_dataset = FUDAN_3d(list_path='./data_list/FUDAN', 
                                  contrast='mr', split='train',crop_size=imsize, train=False, in_ch = 1)

        
    elif dataset == 'ct_fudan':
        
        train_dataset = FUDAN_25d_Nd(list_path='./data_list/FUDAN_2d', 
                                      contrast='ct', split='train',crop_size=imsize, train=True, in_ch = args.in_ch)
            
        test_dataset  = FUDAN_25d(list_path='./data_list/FUDAN_2d', 
                                  contrast='ct', split='train',crop_size=imsize, train=False, in_ch = 1)

        eval_dataset = FUDAN_3d(list_path='./data_list/FUDAN', 
                                  contrast='ct', split='train',crop_size=imsize, train=False, in_ch = 1)


        
    elif dataset == 'sy_t1':
        
        train_dataset = SYNAPSE_25d(list_path='./data_list/SYNAPSE_2d', 
                                      contrast='T1', split='train',crop_size=imsize, train=True, in_ch = args.in_ch)
            
        test_dataset  = SYNAPSE_25d(list_path='./data_list/SYNAPSE_2d', 
                                  contrast='T1', split='train',crop_size=imsize, train=False, in_ch = 1)

        eval_dataset = SYNAPSE_3d(list_path='./data_list/SYNAPSE', 
                                  contrast='T1', split='train',crop_size=imsize, train=False, in_ch = 1)


        
    elif dataset == 'sy_t2':
        
        train_dataset = SYNAPSE_25d(list_path='./data_list/SYNAPSE_2d', 
                                      contrast='T2', split='train',crop_size=imsize, train=True, in_ch = args.in_ch)
            
        test_dataset  = SYNAPSE_25d(list_path='./data_list/SYNAPSE_2d', 
                                  contrast='T2', split='train',crop_size=imsize, train=False, in_ch = 1)

        eval_dataset = SYNAPSE_3d(list_path='./data_list/SYNAPSE', 
                                  contrast='T2', split='train',crop_size=imsize, train=False, in_ch = 1)

        
    elif dataset == 'sy_ct':
        
        train_dataset = SYNAPSE_25d(list_path='./data_list/SYNAPSE_2d', 
                                      contrast='CT', split='train',crop_size=imsize, train=True, in_ch = args.in_ch)
            
        test_dataset  = SYNAPSE_25d(list_path='./data_list/SYNAPSE_2d', 
                                  contrast='CT', split='train',crop_size=imsize, train=False, in_ch = 1)

        eval_dataset = SYNAPSE_3d(list_path='./data_list/SYNAPSE', 
                                  contrast='CT', split='train',crop_size=imsize, train=False, in_ch = 1)




    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=int(workers), pin_memory=True,  drop_last = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=int(workers))
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=int(workers))
    return train_dataloader, test_dataloader , eval_dataloader



def set_converts(datasets, task):
    
    training_converts, test_converts = [], []
    center_dset = datasets[0]
    for source in datasets:  # source
        if not center_dset == source:
            training_converts.append(center_dset + '2' + source)
            training_converts.append(source + '2' + center_dset)
            
        if task=='cmd':
            for target in datasets:  # target
                if not source == target:
                    test_converts.append(source + '2' + target)
                    
    if task  in ['cmd' , 'fudan', 'synapse']:
        tensorboard_converts = test_converts
    else:
        raise Exception("Does not support the task")
    return training_converts, test_converts, tensorboard_converts



