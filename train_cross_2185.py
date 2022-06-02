import sys
import os
import datetime
from torch.serialization import save
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import dataset as data
import data_with_edge_rgbd_2185 as data
from CROSS_rgbd import CROSS
from apex import amp
from tqdm import tqdm

import torchvision

torch.cuda.set_device(3)

def cross_entropy_with_weight(logits, labels):
    logits = logits.sigmoid().view(-1)
    labels = labels.view(-1)
    eps = 1e-6 # 1e-6 is the good choise if smaller than 1e-6, it may appear NaN
    pred_pos = logits[labels > 0].clamp(eps,1.0-eps)
    pred_neg = logits[labels == 0].clamp(eps,1.0-eps)
    w_anotation = labels[labels > 0]

    cross_entropy = (-pred_pos.log() * w_anotation).mean() + \
                    (-(1.0 - pred_neg).log()).mean()

    return cross_entropy

def bce_iou_loss(pred, mask, edge_mask):
    # bce_mask = torch.clamp((1+0.5*edge_mask)*mask,0,1)
    bce   = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou   = 1-(inter+1)/(union-inter+1)

    return (bce+iou).mean()

#/home/pp/Datasets/RGBD_TrainingSet_2185/TrainingSet
#'/home/pp/Datasets/RGBD-TR'
def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='/home/pp/Datasets/RGBD_TrainingSet_2185/TrainingSet', savepath='./models', mode='train_aug', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=32)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=0)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()

    ## parameter
    base, head, base_d = [], [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        elif 'bkbone_d' in name:
            base_d.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}, {'params':base_d}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    # optimizer      = torch.optim.Adam([{'params':base}, {'params':head}], lr=1e-4)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    global_step    = 0

    ck_imgpath = './ckimgs/';
    if not os.path.exists(ck_imgpath):
        os.makedirs(ck_imgpath)
    for epoch in range(cfg.epoch):
        if epoch<=31:
            optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
            optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr
            optimizer.param_groups[2]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        else:
            optimizer.param_groups[0]['lr'] = (1-abs((31+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1*0.5**(epoch-31)
            optimizer.param_groups[1]['lr'] = (1-abs((31+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.5**(epoch-31)
            optimizer.param_groups[2]['lr'] = (1-abs((31+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1*0.5**(epoch-31)

        for step, (image, depth, mask, edge_mask,image_name) in enumerate(tqdm(loader)):
            image,depth ,mask,edge_mask = image.cuda().float(), depth.cuda().float(), mask.cuda().float(), edge_mask.cuda().float()

            sal, edge, e2,e3,e4,e5,g_prior,l_prior = net(image, depth)

            loss_sal = bce_iou_loss(sal, mask,edge_mask)
            loss_edge = bce_iou_loss(edge, mask,edge_mask)
            loss_s2 = cross_entropy_with_weight(e2, edge_mask)
            loss_s3 = bce_iou_loss(e3, mask,edge_mask)
            loss_s4 = cross_entropy_with_weight(e4, edge_mask)
            loss_s5 = bce_iou_loss(e5, mask,edge_mask)

            loss  = 0.5*(loss_sal +  0.2*(loss_edge+loss_s2+loss_s3+loss_s4+loss_s5))

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            
            if step%30==0:

                pred_edge = edge.sigmoid() 
                pred_sal = sal.sigmoid() 
               

                torchvision.utils.save_image(pred_sal, ck_imgpath + 'img'+str(epoch)+'_s.png', nrow=8, padding=2, normalize=False, range=None, scale_each=False)
                

                torchvision.utils.save_image(pred_edge, ck_imgpath + 'img'+str(epoch)+'_e.png', nrow=8, padding=2, normalize=False, range=None, scale_each=False)
                

                torchvision.utils.save_image(mask, ck_imgpath + 'img'+str(epoch)+'_salMask.png', nrow=8, padding=2, normalize=False, range=None, scale_each=False)

                
                torchvision.utils.save_image(edge_mask, ck_imgpath + 'img'+str(epoch)+'_edgeMask.png', nrow=8, padding=2, normalize=False, range=None, scale_each=False)

                
            global_step += 1
            if step%50 == 0:
                # print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))
                # print('loss=%.6f'%(loss.item()))
                # print('loss_edge=%.6f'%(loss_edge.item()))
                # print('loss_sal=%.6f'%(loss_sal.item()))
                # print('\n')

                with open("log.txt","a") as ff:
                    ff.write('%s| epoch %d/%d '%(datetime.datetime.now(), epoch+1, 2*cfg.epoch))
                    ff.write(' | sum_loss=%.6f'%(loss.data))
                    # ff.write(' || s1_loss=%.6f'%(loss1.data))
                    ff.write(' | loss_edge=%.6f'%(loss_edge.data))
                    ff.write(' | loss_sal=%.6f'%(loss_sal.data))
                
                    ff.write('\n')
            
        if (epoch + 1) % 4 == 0 and (epoch+1)>24:
            torch.save(net.state_dict(), cfg.savepath+'/Cross_rgbd_old_cce_2185_' + str(epoch+1) + '.pth')

if __name__=='__main__':
    file = open("log.txt", 'w').close()  #remove content in log.txt
    train(data, CROSS)
