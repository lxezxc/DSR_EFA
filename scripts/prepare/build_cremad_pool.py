#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings

warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random
import re
from collections import defaultdict
from sklearn.metrics import f1_score, average_precision_score
from dsr_efa.config.defaults import config
# from dataset.Samplers import ClassAwareSampler
# from dataset.ClassPrioritySampler import ClassPrioritySampler
from dsr_efa.datasets.CREMA import CramedDataset
# from dataset import create_dataset
# from model.AudioVideo import AudioClassifier, VideoClassifier, AVClassifier, AVEClassifier
from dsr_efa.models.AudioVideo import AVClassifier
# from utils.loss import mixup_criterion
from dsr_efa.utils.utils import (
    create_logger,
    Averager,
    shot_acc,
    deep_update_dict,
    get_optimizer,
    get_scheduler,
    pre_compute_class_ratio,
    freeze_backbone,
    mixup_data,
    lr_reset,
    param_count,
    mixup_data_av
)
def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)

def train_audio_video(epoch, train_loader, model, optimizer, logger,state):
    model.eval()
    
    # ----- RECORD LOSS AND ACC -----
    tl = Averager()
    ta = Averager()
    feats_list, labels_list = [], []
    logits_list = []
    summ = 0
    for step, (spectrogram, image, y) in enumerate(train_loader):

        summ += image.shape[0]
        labels_list.append(torch.argmax(y, dim=1).cpu())
        optimizer.zero_grad()
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        
        if state =='train_audio':
            result_b, result_a, result_v, a_feature, v_feature = model(spectrogram, image)
            f = a_feature
            l = result_a
        elif state =='train_video':
            result_b, result_a, result_v, a_feature, v_feature = model(spectrogram, image)
            f = v_feature
            l = result_v

        elif state == 'train_audio_video_e':
            result_b, result_a, result_v, a_feature, v_feature = model(spectrogram, image)
            # f = v_feature
            # l = result_v
            
            f = a_feature
            l = result_a

        elif state =='train_audio_video':
            # spectrogram, targets_a, targets_b, lam = mixup_data(spectrogram, y, config['train']['mixup_alpha'])
            result_b, result_a, result_v, a_feature, v_feature = model(spectrogram, image)
            f = v_feature
        l = l.detach().float().cpu()
        f = f.detach().float().cpu()
        feats_list.append(f)
        logits_list.append(l)
        optimizer.step()

    logits_raw = torch.cat(logits_list, dim=0).numpy().astype('float32')
    feats_raw = torch.cat(feats_list, dim=0).numpy().astype('float32')
    labels = torch.cat(labels_list, dim=0).numpy().astype('int64')

    denom = np.linalg.norm(feats_raw, axis=1, keepdims=True) + 1e-12
    feats_l2 = (feats_raw / denom).astype('float32')
    print("summ: ", summ)
    return feats_raw, feats_l2, labels, logits_raw

def val(epoch, val_loader, model, logger, state):
    model.eval()
    pred_list = []
    label_list = []
    soft_pred_list = []
    one_hot_label = []
    with torch.no_grad():
        for step, (spectrogram, image, y) in enumerate(val_loader):
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            image = image.cuda()
            y = y.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()
            if state == 'train_audio':
                o = model(spectrogram)
                soft_pred_list = soft_pred_list + F.softmax(o, dim=1).tolist()
                pred_q = F.softmax(o, dim=1).argmax(dim=1)
            elif state == 'train_video':
                o = model(image)
                soft_pred_list = soft_pred_list + F.softmax(o, dim=1).tolist()
                pred_q = F.softmax(o, dim=1).argmax(dim=1)
            elif state == 'train_audio_video_e':
                o = model(spectrogram, image)
                soft_pred_list = soft_pred_list + o.tolist()
                pred_q = F.softmax(o, dim=1).argmax(dim=1)
            elif state == 'train_audio_video':
                o_a, o_v = model(spectrogram, image)
                soft_pred_list = soft_pred_list + ((F.softmax(o_a, dim=1)+F.softmax(o_v, dim=1))/2).tolist()
                pred_q = ((F.softmax(o_a, dim=1)+F.softmax(o_v, dim=1))/2).argmax(dim=1)
            
            pred_list = pred_list + pred_q.tolist()

        f1 = f1_score(label_list, pred_list, average='macro')
        correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
        acc = correct / len(label_list)
        mAP = compute_mAP(torch.Tensor(soft_pred_list), torch.Tensor(one_hot_label))

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('State:{state} Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f}').format(epoch=epoch, f1=f1,
                                                                                                  state=state, acc=acc,
                                                                                                  mAP=mAP))
    return acc

if __name__ == '__main__':

    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cremad_pool.json')
    args = parser.parse_args()
    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    # ----- SET SEED -----
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_id']
    # ----- SET LOGGER -----

    local_rank = cfg['train']['local_rank']
    logger, log_file, exp_id = create_logger(cfg, local_rank)
    # ----- SET DATALOADER -----

    train_dataset = CramedDataset(config, mode='train')
    test_dataset = CramedDataset(config, mode='test')

        
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True, sampler=None,
                              drop_last=True)
    
    ## have 2 version , creamd best is drop last
    # train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
    #                           num_workers=cfg['train']['num_workers'], pin_memory=True, sampler=None,
    #                           drop_last=True)

    
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                             num_workers=cfg['test']['num_workers'], pin_memory=True)

    # ----- MODEL -----
    # state_list = ['train_audio', 'train_video','train_audio_video_e']
    # state_list = ['train_audio', 'train_video']
    state_list = ['train_audio_video_e']
    tips = "logits_our_audio_real"
    
    # state_list = ['train_audio', 'train_video','train_audio_video']
    for state in state_list:
        if state == 'train_audio':
            model = AVClassifier(config=cfg)
            path_audio = "/data/lxe/multimodel/NeurIPS24-LFM-main/BEST_RA_model/Crema_47_best_model_single_audio_best.pth"
            state_dict = torch.load(path_audio, map_location='cuda')
            model.load_state_dict(state_dict, strict=True)
        
        elif state == 'train_video':
            model = AVClassifier(config=cfg)
            path_video = "/data/lxe/multimodel/NeurIPS24-LFM-main/BEST_RA_model/Crema_174_best_model_single_video_best.pth"
            state_dict = torch.load(path_video, map_location='cuda')
            model.load_state_dict(state_dict, strict=True)

        elif state == 'train_audio_video_e':
            model = AVClassifier(config=cfg)
            # path_video = "/data/lxe/multimodel/NeurIPS24-LFM-main/BEST_Creamd_our_Train_RA/Crema_178_best_model.pth"

            # path_video = "/data/lxe/multimodel/NeurIPS24-LFM-main/creamd_model_1117_show_motivation/Static_Crema_video_static_dynamic_best_model_best.pth"
            path_video = "/data/lxe/multimodel/NeurIPS24-LFM-main/creamd_model_1117_show_motivation/Static_Crema_audio_static_dynamic_best_model_best.pth"
            state_dict = torch.load(path_video, map_location='cuda')
            model.load_state_dict(state_dict, strict=True)
        
        elif state == 'train_audio_video':
            model = AVClassifier(config=cfg)
        model = model.cuda()

        lr_adjust = config['train']['optimizer']['lr']
        if config['train']['optimizer']['type'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr_adjust,
                                  momentum=config['train']['optimizer']['momentum'],
                                  weight_decay=config['train']['optimizer']['wc'])
        elif config['train']['optimizer']['type'] == 'ADAM':
            optimizer = optim.Adam(model.parameters(), lr=lr_adjust, betas=(0.9, 0.99),
                                   weight_decay=config['train']['optimizer']['wc'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, config['train']['lr_scheduler']['patience'], 0.1)
        best_acc = 0
        # for epoch in range(cfg['train']['epoch_dict'][state]):
        for epoch in range(1):
            logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))
            scheduler.step()

            ### 
            feats_raw, feats_l2, labels, logits_raw = train_audio_video(epoch, train_loader, model, optimizer, logger, state)

            # print(f"logits_raw {logits_raw.shape}")
            # print(f"feats_raw {feats_raw.shape}")
            # print(f"labels {labels.shape}")
            # assert()

            # return feats_raw, feats_l2, labels, logits_raw
            
            np.save(f'/data/lxe/multimodel/NeurIPS24-LFM-main/appendix/more_embed_show_creamd/BSS/embeddings_raw_{state}_{tips}.npy', feats_raw)
            np.save(f'/data/lxe/multimodel/NeurIPS24-LFM-main/appendix/more_embed_show_creamd/BSS/logits_raw_{state}_{tips}.npy', logits_raw)
            np.save(f'/data/lxe/multimodel/NeurIPS24-LFM-main/appendix/more_embed_show_creamd/BSS/embeddings_l2_{state}_{tips}.npy', feats_l2)
            np.save(f'/data/lxe/multimodel/NeurIPS24-LFM-main/appendix/more_embed_show_creamd/BSS/labels_{state}_{tips}.npy', labels)
            logger.info(f'[Embedding][{state}] raw={feats_raw.shape}, l2={feats_l2.shape}, labels={labels.shape}')

        del model,optimizer,scheduler


