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

class VTModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.hidden = 512
        if config['text']["name"] == 'bert-base':
            self.text_encoder = BertModel.from_pretrained('/data/lxe/multimodel/NeurIPS24-LFM-main/bert-base-uncased', add_pooling_layer=True)

        if config['visual']["name"] == 'resnet50':
            self.visual_encoder = torchvision.models.resnet50()
            checkpoint = torch.load('/data/lxe/multimodel/NeurIPS24-LFM-main/checkpoint/resnet50-0676ba61.pth')
            self.visual_encoder.load_state_dict(checkpoint)

        self.feature_t = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, self.hidden),
        )
        self.feature_i = nn.Sequential(
            nn.Linear(self.visual_encoder.fc.out_features, self.hidden),
        )

        self.cls_t = nn.Linear(self.hidden, config['setting']['num_class'])
        self.cls_i = nn.Linear(self.hidden, config['setting']['num_class'])
    
    # def forward(self, image):
    #     image_embeds = self.feature_i(self.visual_encoder(image))

    #     # text_embeds = self.text_encoder(text.input_ids,
    #     #                                      attention_mask=text.attention_mask,
    #     #                                      return_dict=True
    #     #                                      ).last_hidden_state[:,0,:]
        
    #     # text_embeds = self.feature_t(text_embeds)
    #     f_i = self.cls_i(image_embeds)
    #     # f_t = self.cls_t(text_embeds)
    #     return f_i, image_embeds
    
    def forward(self, image, text):
        image_embeds = self.feature_i(self.visual_encoder(image))

        text_embeds = self.text_encoder(text.input_ids,
                                             attention_mask=text.attention_mask,
                                             return_dict=True
                                             ).last_hidden_state[:,0,:]
        
        text_embeds = self.feature_t(text_embeds)
        f_i = self.cls_i(image_embeds)
        f_t = self.cls_t(text_embeds)
        return f_i, f_t, image_embeds, text_embeds

    def forward_cam(self, image):
        image_embeds = self.feature_i(self.visual_encoder(image))    
        f_i = self.cls_i(image_embeds)

        return f_i, image_embeds
    
    def clssifier(self, image_embed, text_embed, is_image=True, is_text=True):

        if is_image:
            result_i = self.cls_i(image_embed)  # 分别是音频和视频特征经过分类器后得到的预测结果，维度是 [batch_size, num_classes_a] 和 [batch_size, num_classes_v]
        if is_text:
            result_t = self.cls_t(text_embed)
        
        if is_image == True and is_text == False:
            result_v = result_a
        if is_text == True and is_image == False:
            result_a = result_v        

        result_b = result_i
        
        return result_b, result_i, result_t
