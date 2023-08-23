import torch
from dataset import set_converts, get_dataset_with_eval
from models.SDC_UDA import Discriminator, Generator_shallow
from random import seed
from logging import Formatter, StreamHandler, getLogger, FileHandler
import sys
from loss_functions import Loss_Functions
from tqdm import tqdm
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import optim
from tensorboardX import SummaryWriter
import numpy as np
import os
from miou import *
from utils import *
import scipy.io as sio


criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_seg = Diceloss()

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


class Trainer:     
    def __init__(self, args,device):

        self.best_miou = 0.   
        self.best_Dice = 0.
        self.acc = dict()
        self.best_acc = dict()
        self.args= args
        self.device = device
        self.training_converts, self.test_converts, self.tensorboard_converts = set_converts(self.args.datasets, self.args.task)
        self.l2_loss = torch.nn.MSELoss()
        
        
        if args.task in ['cmd', 'fudan','synapse']:
            self.imsize = (args.size, args.size)  
        
                
        for cv in self.test_converts:
            self.best_acc[cv] = 0.

        self.train_loader, self.test_loader, self.data_iter  = dict(), dict(), dict()
        self.train_loader, self.test_loader, self.data_iter, self.eval_loader = dict(), dict(), dict(), dict()
        for dset in self.args.datasets:
            
            train_loader, test_loader, eval_loader = get_dataset_with_eval(dataset=dset, batch=self.args.batch,
                                                    imsize=self.imsize, workers=self.args.N_workers, args=self.args)
            self.train_loader[dset] = train_loader
            self.test_loader[dset] = test_loader
            self.eval_loader[dset] = eval_loader
            
            
            
            
            
            
            
        self.losses = dict()
                
        self.loss_fns = Loss_Functions(args)
        self.writer = SummaryWriter('./tensorboard/%s' % args.ex)
        self.logger = getLogger()
        self.checkpoint = './checkpoint/%s/%s' % (args.task, args.ex)
        self.step = 0
        Tensor = torch.cuda.FloatTensor
        
                
    def set_default(self):
        torch.backends.cudnn.benchmark = True

        ## Random Seed ##
        print("Random Seed: ", self.args.manualSeed)
        print("CUDA_DEVICE  : " , self.args.local_rank)
        seed(self.args.manualSeed)
        torch.manual_seed(self.args.manualSeed)
        torch.cuda.manual_seed_all(self.args.manualSeed)
        self.accum = 0.5 ** (32 / (10 * 1000))

        ## Logger ##
        file_log_handler = FileHandler(self.args.logfile)
        self.logger.addHandler(file_log_handler)
        stderr_log_handler = StreamHandler(sys.stdout)
        self.logger.addHandler(stderr_log_handler)
        self.logger.setLevel('INFO')
        formatter = Formatter()
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)

    
    def save_networks(self):
        if not os.path.exists(self.checkpoint+'/%d' % self.step):
            os.mkdir(self.checkpoint+'/%d' % self.step)

        torch.save(self.netG_A2B_ema.state_dict(),  self.checkpoint + '/%d/netG_A2B_ema.pth'  % (self.step))
        torch.save(self.netG_B2A_ema.state_dict(),  self.checkpoint + '/%d/netG_B2A_ema.pth'  % (self.step))

            

    def load_networks(self, step):
        self.step = step
        
        self.e_ema.load_state_dict(torch.load(self.checkpoint + '/%d/e_ema.pth' % (self.step)))
        self.g_ema.load_state_dict(torch.load(self.checkpoint + '/%d/g_ema.pth' % (self.step)))
        


    def set_models(self):
        
        device =self.device
        
        ##############  Generator  ##################
        generator =Generator_shallow
                
        self.netG_A2B      =   generator(self.args.in_ch , self.args.in_ch, self.args).to(device)
        self.netG_A2B_ema  =   generator(self.args.in_ch , self.args.in_ch, self.args).to(device)

        self.netG_B2A      =   generator(self.args.in_ch , self.args.in_ch, self.args).to(device)
        self.netG_B2A_ema  =   generator(self.args.in_ch , self.args.in_ch, self.args).to(device)
        
                    
        self.netD_A = Discriminator(self.args.in_ch).to(device)
        self.netD_B = Discriminator(self.args.in_ch).to(device)

        self.netG_A2B_ema.eval()
        self.netG_B2A_ema.eval()
        
        accumulate(self.netG_A2B_ema, self.netG_A2B, 0)
        accumulate(self.netG_B2A_ema, self.netG_B2A, 0)
    

    def set_optimizers(self):
        
        self.d_reg_ratio = self.args.d_reg_every / (self.args.d_reg_every + 1)
        
        g_params = list([])
        g_params = g_params + list(self.netG_A2B.parameters())
        g_params = g_params + list(self.netG_B2A.parameters())
        self.g_optim = optim.Adam(  g_params  ,  lr=self.args.lr,  betas=(0, 0.99), )

                
        d_params = list([])
        d_params = d_params + list(self.netD_A.parameters())
        d_params = d_params + list(self.netD_B.parameters())
        self.d_optim = optim.Adam(d_params,lr=self.args.lr * self.d_reg_ratio,  betas=(0 ** self.d_reg_ratio, 0.99 ** self.d_reg_ratio), )



    def get_batch(self, batch_data_iter):
        batch_data = dict()
        for dset in self.args.datasets:
            try:
                batch_data[dset] = batch_data_iter[dset].next()
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_loader[dset])
                batch_data[dset] = batch_data_iter[dset].next()
        return batch_data
    
    
    
    def print_loss(self, pbar):
        best = ''
        if self.args.task in ['cmd'  ,'synapse','fudan']:
            best += '%.2f' % self.best_Dice
        
        losses = ''
        for key in self.losses:
            losses += ('%s: %.2f|'% (key, self.losses[key])) 
        
        pbar.set_postfix({'GPU':self.args.local_rank, 'loss':losses, 'best':best, 'ex':self.args.ex})
    

    def train_step(self, imgs, labels, i ):

        accum = 0.5 ** (32 / (10 * 1000))

#######################################            Train Generator                  #################################################
    
        requires_grads_True([ self.netG_A2B, self.netG_B2A])
        requires_grads_False( [self.netD_A, self.netD_B])
        
        real_imgs = [] ; 
        for keys_i in imgs.keys():       imgs[keys_i].requires_grad = False;    real_imgs.append(imgs[keys_i])    
        
        
        real_A = imgs[self.args.datasets[0]].to(torch.float)
        real_B = imgs[self.args.datasets[1]].to(torch.float)
        
        same_B  = self.netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*1.0
        
        same_A   = self.netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*1.0
        
        # GAN loss
        fake_B  = self.netG_A2B(real_A)
        pred_fake_A = self.netD_B(fake_B)
        loss_GAN_A2B = g_nonsaturating_loss(pred_fake_A[:,0])

        fake_A  = self.netG_B2A(real_B)
        pred_fake_B = self.netD_A(fake_A)
        
        loss_GAN_B2A = g_nonsaturating_loss(pred_fake_B[:,0])
        # Cycle loss

        recovered_A  = self.netG_B2A(fake_B)
        recovered_B  = self.netG_A2B(fake_A)
        
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*3.0
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*3.0
        
        # Image Domain Cyclic Loss
        
        # Seg A Loss

        self.g_optim.zero_grad()
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB 
        
        self.losses['loss_identity_A'] = loss_identity_A.item()
        self.losses['loss_identity_B'] = loss_identity_B.item()
        self.losses['loss_GAN_A2B']    = loss_GAN_A2B.item()
        self.losses['loss_GAN_B2A']    = loss_GAN_B2A.item()
        self.losses['loss_cycle_ABA']  = loss_cycle_ABA.item()
        self.losses['loss_cycle_BAB']  = loss_cycle_BAB.item()
            
        loss_G.backward()
        self.g_optim.step()
        
        accumulate(self.netG_A2B_ema, self.netG_A2B, accum)
        accumulate(self.netG_B2A_ema, self.netG_B2A, accum)


###################################            Train Discriminator                      #############################
        
        requires_grads_False([self.netG_A2B, self.netG_B2A])
        requires_grads_True( [self.netD_A,   self.netD_B])
        
        self.d_optim.zero_grad()

        fake_A = fake_A_buffer.push_and_pop(fake_A.detach())
        pred_real = self.netD_A(real_A)
        pred_fake = self.netD_A(fake_A)
        loss_D_A = d_logistic_loss(pred_real[:,0], pred_fake[:,0])

        
        
        fake_B = fake_B_buffer.push_and_pop(fake_B.detach())
        pred_real = self.netD_B(real_B)
        pred_fake = self.netD_B(fake_B)
        loss_D_B = d_logistic_loss(pred_real[:,0], pred_fake[:,0])
        # Total loss
        
        (loss_D_A+loss_D_B).backward()

        self.d_optim.step()


        self.losses['D_A'] = loss_D_A.item()
        self.losses['D_B'] = loss_D_B.item()


        d_regularize = i % self.args.d_reg_every == 0
        r1_loss = torch.tensor(0, dtype = torch.float, device= self.device)
        
        if d_regularize:
            real_A.requires_grad = True
            real_B.requires_grad = True
        
            
            pred_real = self.netD_A(real_A)
            r1_loss += d_r1_loss(pred_real[:,0], real_A)
            
            pred_real = self.netD_B(real_B)
            r1_loss += d_r1_loss(pred_real[:,0], real_B)
            
            
            self.d_optim.zero_grad()
            r1_loss_sum =  self.args.r1 / 2 * r1_loss * self.args.d_reg_every
            r1_loss_sum.backward()
            self.d_optim.step()

            real_A.requires_grad = False
            real_B.requires_grad = False        
    

    def tensor_board_log(self, imgs, labels):
      
        
        requires_grads_False( [self.netD_A,      self.netD_B])
        requires_grads_False([ self.netG_A2B,  self.netG_B2A])
        
        if self.args.task in ['cmd'  ,'synapse','fudan'] : nrow = 4
        
        real_, fake_, cycle_ = dict(), dict(), dict()
        converts = self.tensorboard_converts
        
        with torch.no_grad():
            
            if self.args.task in ['cmd'  ,'synapse','fudan']: 
            
            
                real_['A'] = imgs[self.args.datasets[0]].to(torch.float)
                real_['B'] = imgs[self.args.datasets[1]].to(torch.float)
                
                fake_['B']  = self.netG_A2B_ema(real_['A'])
                fake_['A']  = self.netG_B2A_ema(real_['B'])
                
                cycle_['A']  = self.netG_B2A_ema(fake_['B'])
                cycle_['B']  = self.netG_A2B_ema(fake_['A'])

            
            
        ######3### WRITE CONVERTED IMAGES #######
            
        # Input Images & Reconstructed Images
        for dset in ['A','B']:
            BB , CC, HH, WW = real_[dset].shape
            x = vutils.make_grid(real_[dset].view(BB*CC, 1, HH, WW).clone().detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('1_Input_Images/%s' % dset, x, self.step)
        for dset in ['A','B']:
            x = vutils.make_grid(fake_[dset].view(BB*CC, 1, HH, WW).clone().detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('2_Fake_Images/%s' % dset, x, self.step)
        # Converted Images
        for convert in ['A','B']:
            x = vutils.make_grid(cycle_[convert].view(BB*CC, 1, HH, WW).clone().detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('3_Cycle_Images/%s' % convert, x, self.step)


        # Losses
        for loss in self.losses.keys():
            self.writer.add_scalar('Losses/%s' % loss, self.losses[loss], self.step)

    
    def eval(self, dset):
        with torch.no_grad():
            
            for batch_idx, (imgs, labels) in enumerate(tqdm(self.eval_loader[dset])):
                imgz  = imgs.cuda().to(torch.float)
                labelz = labels.cuda()
                labelz = labelz.long()
                
                B, C, H, W = imgs.shape
                
                depths_ = int(self.args.in_ch//2)
                img_translated = torch.zeros((B, C, H, W))
            
                for k in range(depths_, C-depths_):
                
                    dummy_in = imgz[:,k-depths_:k+depths_+1]
                    
                    dummy_in = (dummy_in- dummy_in.min()) / (dummy_in.max()-dummy_in.min() + 1e-9)*2-1
                    
                    if dset == self.args.datasets[0]:
                        img_translated[:,k] = self.netG_A2B_ema(dummy_in)[:,depths_][:,None]
                    elif dset == self.args.datasets[1]:
                        img_translated[:,k] = self.netG_B2A_ema(dummy_in)[:,depths_][:,None]


                sio.savemat(self.args.pred_dir + '/step%d_%s_%s.mat' %(self.step,dset,batch_idx),  { "real": imgz[0].permute(1,2,0).detach().cpu().numpy(),
                                                                            "fake": img_translated[0].permute(1,2,0).detach().cpu().numpy(),})
                                                                                            
                                       
    
    def train(self):
        
        print("Setting defaults, models, optimizers...")
    
        self.set_default()
        self.set_models()
        self.set_optimizers()
        
        pbar = range(self.args.iter)
        ## Change this when use multi-GPU
        pbar = tqdm(pbar, initial=self.args.start_iter, dynamic_ncols=True, smoothing=0.01)

        batch_data_iter = dict()
        for dset in self.args.datasets:
            batch_data_iter[dset] = iter(self.train_loader[dset])
            
        batch_data_iter_test = dict()
        for dset in self.args.datasets:
            batch_data_iter_test[dset] = iter(self.test_loader[dset])
            
        for i in pbar:
            self.step += 1
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels = dict(), dict()
            min_batch = self.args.batch
            for dset in self.args.datasets:
                
                imgs[dset], labels[dset] = batch_data[dset]
                imgs[dset], labels[dset] = imgs[dset].to(self.device), labels[dset].to(self.device)
                
                
                
                if self.args.task in ['cmd'  ,'synapse','fudan']:
                    labels[dset] = labels[dset].long()
                if imgs[dset].size(0) < min_batch:
                    min_batch = imgs[dset].size(0)
            if min_batch < self.args.batch:
                for dset in self.args.datasets:
                    imgs[dset], labels[dset] = imgs[dset][:min_batch], labels[dset][:min_batch]
            # training
            
            self.train_step(imgs, labels, i)
            
            # tensorboard
            if self.step % self.args.tensor_freq == 0:
                self.tensor_board_log(imgs, labels)
            
            if self.step % self.args.eval_freq == 0 : 
                
                for dset in self.args.datasets:
                    print('\n DATASET = ', dset, ' STEP = ',self.step, 'START_EVAL')
                    self.eval(dset)
            # evaluation
            if self.step % self.args.eval_freq == 0:
                for cv in self.test_converts:
                    self.eval(cv)
            self.print_loss(pbar)    
            if self.step % self.args.save_freq == 0:
                self.save_networks()
            
    def test(self):
        self.set_default()
        self.set_models()
        self.load_networks(self.args.load_step)
        for cv in self.test_converts:
            self.eval(cv)

            