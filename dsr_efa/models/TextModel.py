from dsr_efa.utils.utils import *
from os import path
from collections import OrderedDict
import torchvision
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict

class TextModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config['text']["name"] == 'bert-base':
            self.text_encoder = BertModel.from_pretrained('/data/lxe/multimodel/NeurIPS24-LFM-main/bert-base-uncased', add_pooling_layer=False)
        elif config['text']["name"] == 'bert-large':
            self.text_encoder = BertModel.from_pretrained('/data/lxe/multimodel/NeurIPS24-LFM-main/bert-large-uncased', add_pooling_layer=False)
        self.hidden_dim = self.text_encoder.config.hidden_size
        self.linear1 = nn.Linear(self.hidden_dim, 512)
        # self.linear2 = nn.Linear(self.hidden_dim, 256)
        # self.linear3 = nn.Linear(self.hidden_dim, 256)
        # self.linear4 = nn.Linear(self.hidden_dim, 256)
        self.cls_t = nn.Sequential(
            # nn.Linear(self.text_encoder.config.hidden_size, 2048),
            # nn.ReLU(),
            # nn.Linear(256, 64),
            nn.Linear(512, config['setting']['num_class'])
        )
    def forward(self, text):
        text_embeds = self.text_encoder(text.input_ids,
                                             attention_mask=text.attention_mask,
                                             return_dict=True
                                             ).last_hidden_state[:,0,:]
        a_feature1 = self.linear1(text_embeds)
        # a_feature2 = self.linear2(text_embeds)
        # a_feature3 = self.linear3(text_embeds)
        # a_feature4 = self.linear4(text_embeds)
        # result = (self.cls_t(a_feature1) + self.cls_t(a_feature2) + self.cls_t(a_feature3) + self.cls_t(
        #     a_feature4)) / 4.0
        # result = (self.cls_t(a_feature1) + self.cls_t(a_feature2)) / 2.0
        result = self.cls_t(a_feature1)
        return result, a_feature1
