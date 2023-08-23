import random 
import os
import time
import sys
import torch
import torch.nn.functional as F
from torch import autograd
from torch import nn
from torch.autograd import Variable



        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))






class Diceloss_heart(nn.Module):
    def __init__(self):
        super(Diceloss_heart, self).__init__()

    def forward(self, y_pred , y_true, smooth=1.):
    
        y_pred_0 = y_pred[:, 0, :, :].contiguous().view(-1)
        y_true_0 = (y_true==0).contiguous().view(-1)
        intersection_0 = (y_pred_0*y_true_0).sum()

        y_pred_1 = y_pred[:, 1, :, :].contiguous().view(-1)
        y_true_1 = (y_true==1).contiguous().view(-1)
        intersection_1 = (y_pred_1*y_true_1).sum()
        
        y_pred_2 = y_pred[:, 2, :, :].contiguous().view(-1)
        y_true_2 = (y_true==2).contiguous().view(-1)
        intersection_2 = (y_pred_2*y_true_2).sum()

        y_pred_3 = y_pred[:, 3, :, :].contiguous().view(-1)
        y_true_3 = (y_true==3).contiguous().view(-1)
        intersection_3 = (y_pred_3*y_true_3).sum()

        y_pred_4 = y_pred[:, 4, :, :].contiguous().view(-1)
        y_true_4 = (y_true==4).contiguous().view(-1)
        intersection_4 = (y_pred_4*y_true_4).sum()

        Dice_BG = (2.*intersection_0 + smooth) / (y_pred_0.sum() + y_true_0.sum() + smooth)
        Dice_1 = (2.*intersection_1 + smooth) / (y_pred_1.sum() + y_true_1.sum() + smooth)
        Dice_2 = (2.*intersection_2 + smooth) / (y_pred_2.sum() + y_true_2.sum() + smooth)
        Dice_3 = (2.*intersection_3 + smooth) / (y_pred_3.sum() + y_true_3.sum() + smooth)
        Dice_4 = (2.*intersection_4 + smooth) / (y_pred_4.sum() + y_true_4.sum() + smooth)
    
        return 1 - (Dice_BG + Dice_1 + Dice_2 + Dice_3 + Dice_4)/5
    
class Diceloss_2ch(nn.Module):
    def __init__(self):
        super(Diceloss_2ch, self).__init__()

    def forward(self, y_pred , y_true, smooth=1.):
    
        y_pred_0 = y_pred[:, 0, :, :].contiguous().view(-1)
        y_true_0 = y_true[:, 0, :, :].contiguous().view(-1)
        intersection_0 = (y_pred_0*y_true_0).sum()

        y_pred_1 = y_pred[:, 1, :, :].contiguous().view(-1)
        y_true_1 = y_true[:, 1, :, :].contiguous().view(-1)
        intersection_1 = (y_pred_1*y_true_1).sum()

        Dice_BG = (2.*intersection_0 + smooth) / (y_pred_0.sum() + y_true_0.sum() + smooth)
        Dice_TUMOR = (2.*intersection_1 + smooth) / (y_pred_1.sum() + y_true_1.sum() + smooth)
    
        return 1 - (Dice_BG + Dice_TUMOR)/2



class Diceloss(nn.Module):
    def __init__(self):
        super(Diceloss, self).__init__()

    def forward(self, y_pred , y_true, smooth=1.):
    
        y_pred_0 = y_pred[:, 0, :, :].contiguous().view(-1)
        y_true_0 = (y_true==0).contiguous().view(-1)
        intersection_0 = (y_pred_0*y_true_0).sum()

        y_pred_1 = y_pred[:, 1, :, :].contiguous().view(-1)
        y_true_1 = (y_true==1).contiguous().view(-1)
        intersection_1 = (y_pred_1*y_true_1).sum()
        
        y_pred_2 = y_pred[:, 2, :, :].contiguous().view(-1)
        y_true_2 = (y_true==2).contiguous().view(-1)
        intersection_2 = (y_pred_2*y_true_2).sum()

        Dice_BG = (2.*intersection_0 + smooth) / (y_pred_0.sum() + y_true_0.sum() + smooth)
        Dice_TUMOR = (2.*intersection_1 + smooth) / (y_pred_1.sum() + y_true_1.sum() + smooth)
        Dice_COCH = (2.*intersection_2 + smooth) / (y_pred_2.sum() + y_true_2.sum() + smooth)
    
        return 1 - (Dice_BG + Dice_TUMOR + Dice_COCH)/3


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()



def accumulate(model1, model2, decay=0.999):
    
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)



# def patchify_image(img, n_crop, min_size=1 / 8, max_size=1 / 4):
#     crop_size = torch.rand(n_crop) * (max_size - min_size) + min_size
#     batch, channel, height, width = img.shape
#     target_h = int(height * max_size)
#     target_w = int(width * max_size)
#     crop_h = (crop_size * height).type(torch.int64).tolist()
#     crop_w = (crop_size * width).type(torch.int64).tolist()

#     patches = []
#     for c_h, c_w in zip(crop_h, crop_w):
#         c_y = random.randrange(0, height - c_h)
#         c_x = random.randrange(0, width - c_w)

#         cropped = img[:, :, c_y : c_y + c_h, c_x : c_x + c_w]
#         cropped = F.interpolate(
#             cropped, size=(target_h, target_w), mode="bilinear", align_corners=False
#         )

#         patches.append(cropped)

#     patches = torch.stack(patches, 1).view(-1, channel, target_h, target_w)

#     return patches




def patchify_image(args, img, n_crop, min_size=1 / 8, max_size=1 / 4):
    
    if args.task == 'clf' : 
        min_size =1/6 ; max_size=1/2
        
    elif args.task == 'brats' : 
        min_size =1/6 ; max_size=1/2
        
    elif args.task == 'seg' : 
        min_size =1/16 ; max_size=1/8
        
    crop_size = torch.rand(n_crop) * (max_size - min_size) + min_size
    
    
    batch, channel, height, width = img.shape
    target_h = int(height * max_size)
    target_w = int(width * max_size)
    crop_h = (crop_size * height).type(torch.int64).tolist()
    crop_w = (crop_size * width).type(torch.int64).tolist()

    patches = []
    for c_h, c_w in zip(crop_h, crop_w):
        if args.task == 'brats' : 
            c_y = random.randrange(c_h, height - c_h)
            c_x = random.randrange(c_w, width - c_w)
        else : 
            c_y = random.randrange(0, height - c_h)
            c_x = random.randrange(0, width - c_w)
        
        cropped = img[:, :, c_y : c_y + c_h, c_x : c_x + c_w]
        cropped = F.interpolate(     cropped, size=(target_h, target_w), mode="bilinear", align_corners=False    )

        patches.append(cropped)

    patches = torch.stack(patches, 1).view(-1, channel, target_h, target_w)
    return patches




def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def requires_grads_True(models):
    for model in models :     
        model.train()
        for p in model.parameters():
            p.requires_grad = True

def requires_grads_False(models):
    for model in models :     
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
            
            
def requires_grads_True_D(models):
    for keys in models.keys() :     
        models[keys].train()
        for p in models[keys].parameters():
            p.requires_grad = True

def requires_grads_False_D(models):
    for keys in models.keys() :     
        models[keys].eval()
        for p in models[keys].parameters():
            p.requires_grad = False
            
            
def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()



def d_r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

