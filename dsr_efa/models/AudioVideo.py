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
from .Resnet import resnet18, resnet34, resnet50


class AudioEncoder(nn.Module):
    def __init__(self, config=None, mask_model=1):
        super(AudioEncoder, self).__init__()
        self.mask_model = mask_model
        if config['text']["name"] == 'resnet18':
            self.audio_net = resnet18(modality='audio')
    
    def forward(self, audio, step=0, balance=0, s=400, a_bias=0):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)  # [512,1]
        a = torch.flatten(a, 1)  # [512]
        return a


class VideoEncoder(nn.Module):
    def __init__(self, config=None, fps=1, mask_model=1):
        super(VideoEncoder, self).__init__()
        self.mask_model = mask_model
        if config['visual']["name"] == 'resnet18':
            self.video_net = resnet18(modality='visual')
        self.fps = fps

    def forward(self, video, step=0, balance=0, s=400, v_bias=0):
        v = self.video_net(video)
        (_, C, H, W) = v.size()
        B = int(v.size()[0] / self.fps)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        return v


class AVClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(AVClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)
        self.hidden_dim = 512
        
        # 可学习的 alpha 参数（用于调整音频和视频原型的修正权重）
        # self.alpha = nn.Parameter(torch.tensor(0.1))  # 初始化为 0.1，训练过程中会自动优化
        
        # 定义一个 全连接层（Linear），用于将输入的特征向量映射到类别空间，进行分类
        self.cls_a = nn.Linear(self.hidden_dim, config['setting']['num_class'])
        self.cls_v = nn.Linear(self.hidden_dim, config['setting']['num_class'])
        # self.cls_b = nn.Linear(self.hidden_dim * 2 , config['setting']['num_class'])
        
    def forward(self, audio, video, audio_run=True, video_run=True):
        
        if audio_run:
            a_feature = self.audio_encoder(audio)  # 通过初始化 AudioEncoder 和 VideoEncoder，分别为音频和视频数据准备特征提取器。
            # 这样，音频和视频数据会通过这些编码器被转化为具有代表性的特征向量，供后续的模型层（如分类头）进行进一步处理  [batch_size, feature_dim]
            result_a = self.cls_a(a_feature)  # 分别是音频和视频特征经过分类器后得到的预测结果，维度是 [batch_size, num_classes_a] 和 [batch_size, num_classes_v]
        
        if video_run:
            v_feature = self.video_encoder(video)
            result_v = self.cls_v(v_feature)
        
        # result_b = self.cls_b(torch.cat((a_feature, v_feature), dim=1))
        # result_b = result_v + result_a
        # result_v = result_a
        # v_feature = a_feature
        result_b = 0
        if audio_run == True and video_run == False:
            v_feature = a_feature
            result_v = result_a

        if video_run == True and audio_run == False:
            a_feature = v_feature
            result_a = result_v
        
        return result_b, result_a, result_v, a_feature, v_feature
        
    def getFeature(self, audio, video, is_audio=True, is_video=True):
        a_feature = 0
        v_feature = 0
        
        if is_audio:
            a_feature = self.audio_encoder(audio)
        if is_video:
            v_feature = self.video_encoder(video)
        return a_feature, v_feature
        

    def clssifier(self, audio_embed, video_embed, is_audio=True, is_video=True):

        if is_audio:
            result_a = self.cls_a(audio_embed)  # 分别是音频和视频特征经过分类器后得到的预测结果，维度是 [batch_size, num_classes_a] 和 [batch_size, num_classes_v]
        if is_video:
            result_v = self.cls_v(video_embed)

        if is_audio == True and is_video == False:
            result_v = result_a
        if is_video == True and is_audio == False:
            result_a = result_v        

        result_b = result_v
        
        return result_b, result_a, result_v



###### new  new
class AudioClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(AudioClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)

        self.hidden_dim = 512
        # self.linear1 = nn.Linear(self.hidden_dim, 256)
        # self.linear2 = nn.Linear(self.hidden_dim, 256)
        # self.linear3 = nn.Linear(self.hidden_dim, 256)
        # self.linear4 = nn.Linear(self.hidden_dim, 256)
        self.cls_a = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, audio):
        a_feature = self.audio_encoder(audio)
        # a_feature1 = self.linear1(a_feature)
        # a_feature2 = self.linear2(a_feature)
        # a_feature3 = self.linear3(a_feature)
        # a_feature4 = self.linear4(a_feature)
        # result_a = (self.cls_a(a_feature1) + self.cls_a(a_feature2)) / 2.0
        # result_a = (self.cls_a(a_feature1) + self.cls_a(a_feature2) + self.cls_a(a_feature3) + self.cls_a(a_feature4)) / 4.0
        result_a = self.cls_a(a_feature)
        # return result_a
        return result_a, a_feature



class VideoClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(VideoClassifier, self).__init__()
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)

        self.hidden_dim = 512
        if config['visual']["name"] == 'resnet50':
            self.hidden_dim = 2048
        # self.linear1 = nn.Linear(self.hidden_dim, 256)
        # self.linear2 = nn.Linear(self.hidden_dim, 256)
        # self.linear3 = nn.Linear(self.hidden_dim, 256)
        # self.linear4 = nn.Linear(self.hidden_dim, 256)
        self.cls_v = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, video):
        v_feature = self.video_encoder(video)
        # v_feature1 = self.linear1(v_feature)
        # v_feature2 = self.linear2(v_feature)
        # v_feature3 = self.linear3(v_feature)
        # v_feature4 = self.linear4(v_feature)
        # result_v = (self.cls_v(v_feature1) + self.cls_v(v_feature2)) / 2.0
        # result_v = (self.cls_v(v_feature1) + self.cls_v(v_feature2) + self.cls_v(v_feature3) + self.cls_v(v_feature4)) / 4.0
        result_v = self.cls_v(v_feature)
        # return result_v
        return result_v, v_feature
