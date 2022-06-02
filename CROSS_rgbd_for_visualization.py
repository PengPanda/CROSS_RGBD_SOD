# from apex.amp.amp import init
# import numpy as np
# import matplotlib.pyplot as plt
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1 ,out2, out3, out4, out5

    def initialize(self):
        res50 = models.resnet50(pretrained=True)
        self.load_state_dict(res50.state_dict(), False)



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class BasicModule(nn.Module):
    def __init__(self, in_channel,out_channel) -> None:
        super(BasicModule, self ).__init__() 
        self.BascBlock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=2, dilation=2), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            # nn.Conv2d(out_channel, out_channel, 3, padding=2, dilation=2), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.BatchNorm2d(out_channel), nn.ReLU(True)
        )

    def forward(self,x):
        x = self.BascBlock(x)
        return  x

    def initialize(self):
        weight_init(self)    

class out_BasicModule(nn.Module):
    def __init__(self, in_channel,out_channel) -> None:
        super(out_BasicModule, self ).__init__() 
        self.BascBlock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, dilation=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.BatchNorm2d(out_channel)
        )

    def forward(self,x):
        x = self.BascBlock(x)
        return  x

    def initialize(self):
        weight_init(self) 

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(2*channel, channel, 3, padding=1, dilation=1), 
            nn.BatchNorm2d(channel), 
            nn.ReLU(True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        y =torch.cat([x, x * y.expand_as(x)],1) 
        y_out = self.conv(y)
        return y_out

    def initialize(self):
        weight_init(self)


class SPLayer_2in(nn.Module):
    def __init__(self, chnl_1, chnl_2, outchnl=64):
        super(SPLayer_2in, self).__init__()
        self.W_g = nn.Sequential(
            conv1x1(chnl_1, outchnl),
            nn.BatchNorm2d(outchnl)
        )
        self.W_x = nn.Sequential(
            conv1x1(chnl_2, outchnl),
            nn.BatchNorm2d(outchnl)
        )

        self.spatial_attention = nn.Sequential(
            conv1x1(outchnl, 1),
            nn.BatchNorm2d(1),
            # nn.Tanh()
            # nn.Sigmoid()
        )  # spatial saliency
        self.relu = nn.ReLU(inplace=True)
        self.seb = SELayer(outchnl, reduction=1)

    def forward(self, x1, x2):
        x1 = self.W_g(x1)
        x2 = self.W_x(x2)

        x12 = self.relu(x1 + x2)
        x12 = self.seb(x12)
        sa_map = self.spatial_attention(x12)

        return x12,sa_map

    def initialize(self):
        weight_init(self)



class Old_CCE(nn.Module):
    def __init__(self, in_chnl1,in_chnl2, out_chnl):
        super(Old_CCE, self ).__init__()
        self.basicConv_a = BasicModule(in_chnl1, out_chnl)
        self.basicConv_b = BasicModule(in_chnl2, out_chnl)
        self.spb = SPLayer_2in(out_chnl, out_chnl, out_chnl)
        self.basicConv_fa = BasicModule(2*out_chnl, out_chnl)
        self.basicConv_fb = BasicModule(2*out_chnl, out_chnl)
        self.basicConv_fab = BasicModule(2*out_chnl, out_chnl)

        self.spatial_attention = nn.Sequential(
            conv1x1(out_chnl, 1),
            nn.BatchNorm2d(1),
            # nn.Tanh()
        )  # spatial saliency
        
    def forward(self, xc, xd, prior):
        xa = self.basicConv_a(xc)
        xb = self.basicConv_b(xd)
        if xa.shape[2] < xb.shape[2]:
            xa = F.interpolate(xa, mode='bilinear',scale_factor=2,align_corners=True)
        elif xb.shape[2] < xa.shape[2]:
            xb = F.interpolate(xb, mode='bilinear',scale_factor=2,align_corners=True)

        xab,sa = self.spb(xa,xb)

        sa =F.sigmoid(sa**2) 
        prior = F.sigmoid(prior**2)
        att_map = torch.max(torch.cat((sa,prior),1),dim=1, keepdim=True)[0]
        rev_attmap = 1 - att_map 
        # rev_attmap =torch.sub(torch.ones_like(att_map,requires_grad=False),att_map) 


        xa2 = self.basicConv_fa(torch.cat((att_map*xa, rev_attmap*xa),1))
        xb2 = self.basicConv_fb(torch.cat((att_map*xb, rev_attmap*xb),1))
        xab2 = self.basicConv_fab(torch.cat((att_map*xab, rev_attmap*xab),1))

        xf = xa2 + xb2 + xab2
        prior_map = self.spatial_attention(xf)

        return xf, prior_map, sa

    def initialize(self):
        weight_init(self)



class sal_decoder(nn.Module):
    def __init__(self, out_channel):
        super(sal_decoder, self).__init__() 
        self.BasicConv_5 = BasicModule(2048, out_channel)
        self.BasicConv_51 = BasicModule(3*out_channel, out_channel)

        self.BasicConv_4 = BasicModule(out_channel, out_channel)
        self.BasicConv_41 = BasicModule(3*out_channel, out_channel)

        self.BasicConv_3 = BasicModule(out_channel, out_channel)
        self.BasicConv_31 = BasicModule(3*out_channel, out_channel)

        self.BasicConv_2 = BasicModule(out_channel, out_channel)
        self.BasicConv_21 = BasicModule(3*out_channel, out_channel)

        self.BasicConv_1 = out_BasicModule(out_channel, 1)
        self.BasicConv_11 = BasicModule(3*out_channel, out_channel)


    def forward(self, x5, e1, e2, e3,e4, e5, f1,f2,f3, f4, f5):
        m5 = self.BasicConv_5(x5)   #out = 64
        m5 = self.BasicConv_51(torch.cat([m5,e5,f5],1))  #out = 64
        m5_up = F.interpolate(m5, scale_factor=2, mode='bilinear', align_corners=True) 

        m4 = self.BasicConv_4(m5_up)   #out = 64
        m4 = self.BasicConv_41(torch.cat([m4,e4,f4],1))  #out = 64
        m4_up = F.interpolate(m4, scale_factor=2, mode='bilinear', align_corners=True) 

        m3 = self.BasicConv_3(m4_up)   #out = 64
        m3 = self.BasicConv_31(torch.cat([m3,e3,f3],1))  #out = 64
        m3_up = F.interpolate(m3, scale_factor=2, mode='bilinear', align_corners=True) 

        m2 = self.BasicConv_2(m3_up)   #out = 64
        m2 = self.BasicConv_21(torch.cat([m2,e2,f2],1))  #out = 64
        # m2_up = F.interpolate(m2, scale_factor=2, mode='bilinear', align_corners=True) 

        m1 = self.BasicConv_11(torch.cat([m2,e1,f1],1))  #out = 64

        sal_out = self.BasicConv_1(m1)   #out = 64

        return sal_out,m2,m3,m4,m5
  
    




class depth_decoder(nn.Module):
    def __init__(self, out_channel):
        super(depth_decoder, self).__init__() 
        self.BasicConv_5 = BasicModule(2048, out_channel)
        self.BasicConv_51 = BasicModule(2*out_channel, out_channel)

        self.BasicConv_4 = BasicModule(out_channel, out_channel)
        self.BasicConv_41 = BasicModule(2*out_channel, out_channel)

        self.BasicConv_3 = BasicModule(out_channel, out_channel)
        self.BasicConv_31 = BasicModule(2*out_channel, out_channel)

        self.BasicConv_2 = BasicModule(out_channel, out_channel)
        self.BasicConv_21 = BasicModule(2*out_channel, out_channel)

        self.BasicConv_1 = out_BasicModule(out_channel, 1)
        self.BasicConv_11 = BasicModule(2*out_channel, out_channel)



    def forward(self, x5, f1,f2,f3, f4, f5):
        m5 = self.BasicConv_5(x5)   #out = 64
        m5 = self.BasicConv_51(torch.cat([m5,f5],1))  #out = 64
        m5_up = F.interpolate(m5, scale_factor=2, mode='bilinear', align_corners=True) 

        m4 = self.BasicConv_4(m5_up)   #out = 64
        m4 = self.BasicConv_41(torch.cat([m4,f4],1))  #out = 64
        m4_up = F.interpolate(m4, scale_factor=2, mode='bilinear', align_corners=True) 

        m3 = self.BasicConv_3(m4_up)   #out = 64
        m3 = self.BasicConv_31(torch.cat([m3,f3],1))  #out = 64
        m3_up = F.interpolate(m3, scale_factor=2, mode='bilinear', align_corners=True) 

        m2 = self.BasicConv_2(m3_up)   #out = 64
        m2 = self.BasicConv_21(torch.cat([m2,f2],1))  #out = 64


        m1 = self.BasicConv_11(torch.cat([m2,f1],1))  #out = 64

        edge_out = self.BasicConv_1(m1)   #out = 64

        return edge_out, m1, m2, m3, m4, m5



class CROSS(nn.Module):
    def __init__(self, cfg, channel=64):
        self.cfg = cfg
        super(CROSS, self).__init__()
        self.bkbone = ResNet()
        self.bkbone_d = ResNet()
         ## prior
        self.get_prior = nn.Sequential(
            conv1x1(2048, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )  # spatial saliency

        self.cce5 = Old_CCE(2048,2048,channel)
        self.cce4 = Old_CCE(1024,1024,channel)
        self.cce3 = Old_CCE(512,512,channel)
        self.cce2 = Old_CCE(256,256,channel)
        self.cce1 = Old_CCE(64,64,channel)

        self.sal_decoder = sal_decoder(64)
        self.depth_decoder = depth_decoder(64)

        self.conv2 = out_BasicModule(channel,1)
        self.conv3 = out_BasicModule(channel,1)
        self.conv4 = out_BasicModule(channel,1)
        self.conv5 = out_BasicModule(channel,1)

        self.initialize()


    def forward(self, x, xd):
        x1, x2, x3, x4, x5 = self.bkbone(x)
        d1, d2, d3, d4, d5 = self.bkbone_d(xd)
        x_size = x.size()[2:]

        prior6 = self.get_prior(x5+d5)

        f5, prior5, sa5 = self.cce5(x5,d5,prior6)
        prior5 = F.interpolate(prior5,scale_factor=2,mode='bilinear',align_corners=True)

        f4, prior4, sa4 = self.cce4(x4,d4,prior5)
        prior4 = F.interpolate(prior4,scale_factor=2,mode='bilinear',align_corners=True)

        f3, prior3, sa3 = self.cce3(x3,d3,prior4)

        prior5 = F.interpolate(prior5,scale_factor=4,mode='bilinear',align_corners=True)
        prior4 = F.interpolate(prior4,scale_factor=2,mode='bilinear',align_corners=True)
        prior3 = F.interpolate(prior3,scale_factor=2,mode='bilinear',align_corners=True)
        global_prior = (prior3 + prior4 + prior5)/3

        f2, prior2, sa2 = self.cce2(x2,d2,global_prior)
        # global_prior = F.interpolate(global_prior,scale_factor=2,mode='bilinear',align_corners=True)
        f1, prior1, sa1 = self.cce1(x1,d1,global_prior)

        # prior2 = F.interpolate(prior2,scale_factor=2,mode='bilinear',align_corners=True)
        local_prior = (prior1+prior2)/2

        edge, e1,e2,e3,e4,e5 = self.depth_decoder(d5,f1,f2,f3,f4,f5)
        sal,m2,m3,m4,m5 = self.sal_decoder(x5,e1,e2,e3,e4,e5,f1,f2,f3,f4,f5)

        s2 = self.conv2(m2)
        ed2 = F.interpolate(s2, x_size, mode='bilinear', align_corners=True) 
        s3 = self.conv3(m3)
        ed3 = F.interpolate(s3, x_size, mode='bilinear', align_corners=True) 
        s4 = self.conv4(m4)
        ed4 = F.interpolate(s4, x_size, mode='bilinear', align_corners=True) 
        s5 = self.conv5(m5)
        ed5 = F.interpolate(s5, x_size, mode='bilinear', align_corners=True)

        edge_out =  F.interpolate(edge, x_size, mode='bilinear', align_corners=True) 
        sal_out = F.interpolate(sal, x_size, mode='bilinear', align_corners=True) 

        pmap = []
        pmap.append(prior1)
        pmap.append(prior2)
        pmap.append(prior3)
        pmap.append(prior4)
        pmap.append(prior5)
        # pmap.append(prior6)

        sa = []
        sa.append(sa1)
        sa.append(sa2)
        sa.append(sa3)
        sa.append(sa4)
        sa.append(sa5)

        depth_fea = []
        depth_fea.append(d1)
        depth_fea.append(d2)
        depth_fea.append(d3)
        depth_fea.append(d4)
        depth_fea.append(d5)

        rgb_fea = []
        rgb_fea.append(x1)
        rgb_fea.append(x2)
        rgb_fea.append(x3)
        rgb_fea.append(x4)
        rgb_fea.append(x5)


        return  sal_out, edge_out, ed2,ed3,ed4,ed5, global_prior, local_prior,pmap,sa,depth_fea,rgb_fea
       

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
            print('model loaded')
        else:
            weight_init(self)







if __name__=='__main__':
    model = CROSS( cfg=1,channel=64)
    print(model)

    input1 = torch.randn(8, 3, 256, 256)
    input2 = torch.randn(8, 3, 256, 256)
    out = model(input1,input2)
    print(out.shape)