from os import path
from collections import OrderedDict
import torchvision
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict

class VisualModel(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        #download the bert-base from huggingface
        # https://huggingface.co/google-bert/bert-base-uncased/tree/main
        #download the resnet from torchvision
        # https://download.pytorch.org/models/resnet50-0676ba61.pth
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        if config['visual']["name"] == 'resnet18':
            self.visual_encoder = torchvision.models.resnet18()
            checkpoint = torch.load('/data/lxe/multimodel/NeurIPS24-LFM-main/checkpoint/resnet18-f37072fd.pth')
            self.visual_encoder.load_state_dict(checkpoint)
        elif config['visual']["name"] == 'resnet34':
            self.visual_encoder = torchvision.models.resnet34()
            checkpoint = torch.load('/data/lxe/multimodel/NeurIPS24-LFM-main/checkpoint/resnet34-333f7ec4.pth')
            self.visual_encoder.load_state_dict(checkpoint)
        elif config['visual']["name"] == 'resnet50':
            self.visual_encoder = torchvision.models.resnet50()
            checkpoint = torch.load('/data/lxe/multimodel/NeurIPS24-LFM-main/checkpoint/resnet50-0676ba61.pth')
            self.visual_encoder.load_state_dict(checkpoint)
        self.hidden_dim = self.visual_encoder.fc.out_features
        self.linear1 = nn.Linear(self.hidden_dim, 512)
        # self.linear2 = nn.Linear(self.hidden_dim, 256)
        # self.linear3 = nn.Linear(self.hidden_dim, 256)
        # self.linear4 = nn.Linear(self.hidden_dim, 256)

        # self.cls_i = nn.Sequential(
        #     # nn.Linear(self.visual_encoder.fc.out_features, 2048),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.Linear(64, config['setting']['num_class'])
        # )
        
        self.cls_i = nn.Sequential(
            # nn.Linear(self.visual_encoder.fc.out_features, 2048),
            # nn.ReLU(),
            # nn.Linear(256, 64),
            nn.Linear(512, config['setting']['num_class'])
        )
        
    def forward(self, image):
        image_embeds = self.visual_encoder(image)
        a_feature1 = self.linear1(image_embeds)
        # a_feature2 = self.linear2(image_embeds)
        # a_feature3 = self.linear3(image_embeds)
        # a_feature4 = self.linear4(image_embeds)
        # result = (self.cls_i(a_feature1) + self.cls_i(a_feature2) + self.cls_i(a_feature3) + self.cls_i(
        #     a_feature4)) / 4.0
        # result = (self.cls_i(a_feature1) + self.cls_i(a_feature2)) / 2.0
        result = self.cls_i(a_feature1)
        return result, a_feature1