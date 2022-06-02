import os
import sys
sys.dont_write_bytecode = True
import cv2
import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import data_with_edge_rgbd as data
from CROSS_rgbd  import CROSS
import time

torch.cuda.set_device(1)

class Test(object):
    def __init__(self, Dataset, Network, dataset_name):
        ## dataset
        data_root   = './'
        self.path   = os.path.join(data_root, dataset_name, 'test_data')
        self.cfg    = Dataset.Config(datapath=self.path, snapshot='./models/Cross_rgbd_32.pth', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.dataset_name = dataset_name
        
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
    
    def save(self):
        with torch.no_grad():
            time_t = 0.0+1e-6

            for image,depth, shape, name in self.loader:
                image,depth = image.cuda().float(),depth.cuda().float()
                time_start = time.time()
                res,_,_,_, _,_,_,_= self.net(image,depth)
                torch.cuda.synchronize()
                time_end = time.time()
                time_t = time_t + time_end - time_start
                res = F.interpolate(res, shape, mode='bilinear', align_corners=True)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = 255 * res
                save_path  = os.path.join('./vis_feature/show_images/')  # visualize  your feature maps
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path+'/'+name[0]+'_lp.png', res)
            fps = len(self.loader) / time_t
            print('FPS is %f' %(fps))


if __name__=='__main__':
    for data_path in ['vis_feature']: 
    # for data_path in ['SIP']:
        test = Test(data, CROSS, data_path)
        test.save()
