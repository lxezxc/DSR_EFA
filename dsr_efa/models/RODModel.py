from dsr_efa.utils.utils import *
from os import path
import torchvision
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from dsr_efa.models.Resnet import resnet18

import torch.nn as nn
import torch
from dsr_efa.models.Model3D import InceptionI3d

class RGBEncoder(nn.Module):
    def __init__(self, config):
        super(RGBEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=3)
        #download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/lxe/multimodel/NeurIPS24-LFM-main/checkpoint/rgb_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.rgbmodel = model

    def forward(self, x):
        out = self.rgbmodel(x)
        return out  # BxNx2048


class OFEncoder(nn.Module):
    def __init__(self, config):
        super(OFEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=2)
        #download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/lxe/multimodel/NeurIPS24-LFM-main/checkpoint/flow_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.ofmodel = model

    def forward(self, x):
        out = self.ofmodel(x)
        return out  # BxNx2048


class DepthEncoder(nn.Module):
    def __init__(self, config):
        super(DepthEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=1)
        #download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/lxe/multimodel/NeurIPS24-LFM-main/checkpoint/rgb_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.depthmodel = model

    def forward(self, x):
        out = self.depthmodel(x)
        return out  # BxNx2048

class RGBClsModel(nn.Module):
    def __init__(self, config):
        super(RGBClsModel, self).__init__()
        self.rgb_encoder = RGBEncoder(config)

        self.hidden_dim = 1024
        self.cls_r = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1):
        rgb = x1
        rgb_feat = self.rgb_encoder(rgb)
        result_r = self.cls_r(rgb_feat)
        return result_r, rgb_feat


class OFClsModel(nn.Module):
    def __init__(self, config):
        super(OFClsModel, self).__init__()
        self.of_encoder = OFEncoder(config)

        self.hidden_dim = 1024
        self.cls_o = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1):
        of = x1
        of_feat = self.of_encoder(of)
        result_o = self.cls_o(of_feat)
        return result_o, of_feat

class DepthClsModel(nn.Module):
    def __init__(self, config):
        super(DepthClsModel, self).__init__()
        self.depth_encoder = DepthEncoder(config)

        self.hidden_dim = 1024
        self.cls_d = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1):
        depth = x1
        depth_feat = self.depth_encoder(depth)
        result_d = self.cls_d(depth_feat)
        return result_d, depth_feat

class JointClsModel(nn.Module):
    def __init__(self, config):
        super(JointClsModel, self).__init__()
        self.rgb_encoder = RGBEncoder(config)
        self.of_encoder = OFEncoder(config)
        self.depth_encoder = DepthEncoder(config)
        self.hidden_dim = 1024
        self.cls_r = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )
        self.cls_o = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )
        self.cls_d = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1, x2, x3):
        rgb = x1
        of = x2
        depth = x3

        rgb_feat = self.rgb_encoder(rgb)
        result_r = self.cls_r(rgb_feat)

        of_feat = self.of_encoder(of)
        result_o = self.cls_o(of_feat)


        depth_feat = self.depth_encoder(depth)
        result_d = self.cls_d(depth_feat)

        return result_r, result_o, result_d, rgb_feat, of_feat, depth_feat

    def clssifier(self, r_embed, o_embed, d_embed):
        result_r = self.cls_r(r_embed)
        result_o = self.cls_o(o_embed)
        result_d = self.cls_d(d_embed)
        return result_r, result_o, result_d
