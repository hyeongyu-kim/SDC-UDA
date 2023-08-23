from __future__ import print_function
import torch
from args import get_args
from trainer import Trainer


if __name__ == '__main__':
    opt = get_args()
    

    device = "cuda"
    torch.cuda.set_device(opt.local_rank)
    trainer = Trainer(opt,device)
    trainer.train()
