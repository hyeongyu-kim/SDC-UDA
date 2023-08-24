import argparse
import os
from re import S

def check_dirs(dirs):
    dirs = [dirs] if type(dirs) not in [list, tuple] else dirs
    for d in dirs:
        try:
            os.makedirs(d)
        except OSError:
            pass
    return


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str , default = 'cmd')    
    parser.add_argument("--dataset_used", type=str, default= 'CrossMODA' )
    parser.add_argument("--datasets", type=int, default= ['to','tt'] ) #T1 : to , T2 : tt 
    parser.add_argument("--N_seg_cls", type=int, default= 3 ) # VS, C, bg
    
    # parser.add_argument("--task", type=str , default = 'fudan')    
    # parser.add_argument("--dataset_used", type=str, default= 'FUDAN' )
    # parser.add_argument("--datasets", type=int, default= ['mr_fudan','ct_fudan'] ) #
    # parser.add_argument("--N_seg_cls", type=int, default= 5 ) #
    
    
    # parser.add_argument("--task", type=str , default = 'synapse')    
    # parser.add_argument("--dataset_used", type=str, default= 'SYNAPSE' )
    # parser.add_argument("--datasets", type=int, default= ['sy_t1','sy_t2'] ) #
    # parser.add_argument("--N_seg_cls", type=int, default= 5 ) #
    


    parser.add_argument("--local_rank",         type=int, default=0)   # GPU ID
    parser.add_argument("--ex", type=str, default= 'Your_Experiment_Name') ## Every experiment

    parser.add_argument("--in_ch",                type=int,      default=3)
    
    parser.add_argument("--save_freq",           type=int,    default=10000) 
    parser.add_argument("--tensor_freq",         type=int,    default=500) 
    parser.add_argument("--eval_freq",           type=int,    default=10000) 

    parser.add_argument("--lr", type=float,       default=5e-5) 
     
    parser.add_argument('--logfile',              type=str)
    
    parser.add_argument("--N_workers",             type=int,      default = 8)
    parser.add_argument('--manualSeed',            type=int,      default = 5688)        
    parser.add_argument("--iter",                  type=int,      default = 100000)
    parser.add_argument("--batch",                 type=int,      default = 1)  ## Recommend batch 1, or need to modify 'EfficientSelfAttention' somewhere. 
    parser.add_argument("--r1",                    type=float,    default = 10)
    parser.add_argument("--d_reg_every",           type=int,      default=12)
    parser.add_argument("--start_iter",            type=int,      default=0)
    
    parser.add_argument('--weight_decay_task',     type=float, default=5e-4)
    parser.add_argument("--ckpt",                  type=str, default=None)
    
    parser.add_argument("--channel", type=int, default=32)
    
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    
    args = parser.parse_args()

    check_dirs(['checkpoint/' + args.task + '/' + args.ex])
    args.logfile = './checkpoint/' + args.task + '/' + args.ex + '/' + args.ex + '.log'
    
    check_dirs(['preddir/' + '/' + args.ex])
    args.pred_dir = './preddir/' + '/' + args.ex
    
    
    args.size= 256
    # args.batch = 1

    return args


        
        
        
        
        
        
    
    
