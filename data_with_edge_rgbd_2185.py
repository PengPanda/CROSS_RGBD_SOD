# import os
from unicodedata import name
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, depth, mask=None, edge_mask=None):
        image = (image - self.mean)/self.std
        depth = (depth - self.mean)/self.std
        if mask is None:
            return image,depth
        else:
            mask /= 255
            edge_mask /=255
            
            return image, depth, mask, edge_mask

class RandomCrop(object):
    def __call__(self, image, mask, edge_mask):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3], edge_mask[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask, edge_mask):
        if np.random.randint(2)==0:
            return image[:,::-1,:], mask[:, ::-1], edge_mask[:, ::-1]
        else:
            return image, mask, edge_mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image,depth, mask=None, edge_mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)

        if mask is None:
            return image,depth
        else:
            mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            edge_mask  = cv2.resize( edge_mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, depth, mask, edge_mask

class ToTensor(object):
    def __call__(self, image,depth, mask=None, edge_mask = None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        depth = torch.from_numpy(depth)
        depth = depth.permute(2, 0, 1)

        if mask is None:
            return image,depth
        else:
            mask  = torch.from_numpy(mask)
            edge_mask  = torch.from_numpy(edge_mask)
            return image, depth, mask, edge_mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(256, 256)
        self.totensor   = ToTensor()
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())
        # self.samples = self.samples[:32] #testfortrain

    def __getitem__(self, idx):
        name  = self.samples[idx]
        

        if self.cfg.mode=='train_aug':
            image = cv2.imread(self.cfg.datapath+'/aug_images/'+name+'.jpg')[:,:,::-1].astype(np.float32)
            shape = image.shape[:2]
            depth = cv2.imread(self.cfg.datapath+'/aug_depths/'+name+'.png', 1).astype(np.float32)
            mask  = cv2.imread(self.cfg.datapath+'/aug_masks/' +name+'.png', 0).astype(np.float32)
            edge_mask = cv2.imread(self.cfg.datapath+'/aug_edges/' +name+'.png', 0).astype(np.float32)

            image, depth, mask, edge_mask = self.normalize(image,depth, mask, edge_mask)
            # image, mask, edge_mask = self.randomcrop(image, mask, edge_mask)
            # image, mask, edge_mask = self.randomflip(image, mask, edge_mask)
            return image,depth, mask, edge_mask,name
        else:
            image = cv2.imread(self.cfg.datapath+'/images/'+name+'.jpg')[:,:,::-1].astype(np.float32)
            shape = image.shape[:2]
            depth = cv2.imread(self.cfg.datapath+'/depths/'+name+'.png', 1).astype(np.float32)
            image,depth = self.normalize(image,depth)
            image,depth = self.resize(image,depth)            
            image,depth = self.totensor(image,depth)
            
            return image,depth, shape, name

    def collate(self, batch):
        # size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        size = 256
        image,depth, mask, edge_mask,name = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            depth[i]  = cv2.resize(depth[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            edge_mask[i]  = cv2.resize(edge_mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)

        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        depth = torch.from_numpy(np.stack(depth, axis=0)).permute(0,3,1,2)
        mask  = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        edge_mask  = torch.from_numpy(np.stack(edge_mask, axis=0)).unsqueeze(1)
        
        return image, depth, mask, edge_mask, name

    def __len__(self):
        return len(self.samples)
