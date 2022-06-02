import torch
import cv2
import os.path
import numpy as np
from torch import nn
from tqdm import tqdm

import os
import sys
sys.dont_write_bytecode = True
import torch.nn.functional as F
from torch.utils.data import DataLoader
import data_with_edge_rgbd as data
from CROSS_rgbd_for_visualization  import CROSS
import time

torch.cuda.set_device(0)

class Visualize(object):
    def __init__(self,save_path) -> None:
        super(Visualize).__init__()
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    def forward(self,features,name):

        # mean_fea = []
        for i in range(len(features)):
            fea = features[i]
            fea_img = torch.mean(torch.abs(fea),dim=1)
            fea_img = fea_img.squeeze().cpu().numpy() #cpu()
            fea_img = cv2.GaussianBlur(fea_img,(15,15),0)
            fea_img = self.normlize(fea_img)
            fea_img = 255*cv2.resize(fea_img, (256,256))
            fea_img = cv2.applyColorMap(fea_img.astype(np.uint8),cv2.COLORMAP_JET)
        
            img_name = self.save_path +name  + '_fea_' + str(i+1) + '.png'
            cv2.imwrite(img_name,fea_img)



        # mean_fea_img = np.mean(mean_fea,axis=0)
        # # mean_fea_img = cv2.GaussianBlur(mean_fea_img,(9,9),0)
        # mean_fea_img = cv2.applyColorMap(mean_fea_img.astype(np.uint8),cv2.COLORMAP_JET)
        # img_name = self.save_path + name +'_' + '_meanfea_' + '.png'
        # cv2.imwrite(img_name,mean_fea_img)

        # return fea_img
    
    def normlize(self,x):
        xmax = x.max()
        xmin = x.min()
        res = (x-xmin)/(xmax-xmin+0.000001)

        return res

def save_res(res_list,shape,name, save_path, key_word = 'prior',mod='Neg'):
    for i,res in enumerate(res_list):
        res = F.interpolate(res, shape, mode='bilinear', align_corners=True)
        res = torch.mean(torch.abs(res),1)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        if mod == 'Neg':
            res = 255 * (1-res)
        elif mod == 'Pos':
            res = 255 * (res)
        if i>=2:
            res = cv2.GaussianBlur(res,(15,15),0)
        else:
            res = cv2.GaussianBlur(res,(51,51),0)
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        res = cv2.applyColorMap(res.astype(np.uint8),cv2.COLORMAP_JET)

        # save_path  = os.path.join('./vis_feature/show_images/', dataset_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cv2.imwrite(save_path+'/'+name[0] +'_' + key_word +'_'+ str(i+1) + '_.png', res)
    



class Test(object):
    def __init__(self, Dataset, Network, dataset_name):
        ## dataset
        data_root   = './vis_feature/'
        self.path   = os.path.join(data_root, dataset_name)
        self.cfg    = Dataset.Config(datapath=self.path, snapshot='./models/Cross_rgbd_32.pth', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.dataset_name = dataset_name
        
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

        self.save_vis = Visualize(data_root)
    
    def save(self):
        with torch.no_grad():
            time_t = 0.0+1e-6

            for idx, (image,depth, shape, name) in enumerate(tqdm(self.loader)):
                image,depth = image.cuda().float(),depth.cuda().float()
                time_start = time.time()
                res1,_,_,_, _,_,gp,lp,fea, sa, depth_fea,rgb_fea= self.net(image,depth)
                torch.cuda.synchronize()
                time_end = time.time()
                time_t = time_t + time_end - time_start
                # self.save_vis.forward(fea,name[0])   #save feature

                save_path = os.path.join('./vis_feature/show_images/','test_data')
                # save_res(gp,shape,name,save_path,key_word='global_prior')
                # save_res(lp,shape,name,save_path,key_word='local_prior')

                save_res(fea,shape,name,save_path,key_word='prior',mod='Pos')
                save_res(sa,shape,name,save_path,key_word='sa',mod='Pos')
                save_res(depth_fea,shape,name,save_path,key_word='depth_fea',mod='Pos')
                save_res(rgb_fea,shape,name,save_path,key_word='rgb_fea',mod='Pos')

               


            fps = len(self.loader) / time_t
            print('FPS is %f' %(fps))


if __name__=='__main__':
    # data_root = '/home/pp/WorkSpace/PythonSpace/pytorch/Datasets/'
    for data_path in ['test_data']: # 'ECSSD', 'PASCAL', 'HKUIS', 'DUTS-TE', 'DUTO'        
    # for data_path in ['SIP']:
        test = Test(data, CROSS, data_path)
        test.save()
    
