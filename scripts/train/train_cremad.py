#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from tqdm import tqdm
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random
import re
from collections import defaultdict
from sklearn.metrics import f1_score, average_precision_score
from dsr_efa.config.defaults import config
from dsr_efa.datasets.CREMA import CramedDataset
from torch.utils.data import DataLoader, BatchSampler, Subset
import time

from dsr_efa.models.AudioVideo import AVClassifier
from dsr_efa.utils.loss import mixup_criterion
from datetime import datetime
import math
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
from dsr_efa.utils.tools import GSPlugin, weight_init
# import setproctitle
# setproctitle.setproctitle("lxe")

def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)

## o_a, o_v                     --->   feature to search
## score_a_data, score_v_data   --->   sample number that need

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x

def _as_2d(x):
    x = _to_numpy(x)
    if x.ndim == 1:
        x = x[None, :]
    return x  # [B, D]

def _ensure_l2(X, eps=1e-12):
    X = _to_numpy(X)
    if X.ndim == 1:
        X = X[None, :]
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (n + eps)
    return X.astype(np.float32, copy=False)

def _topk_sim_batch_top(queries, db_emb, k):
    """
    queries: [B, D]
    db_emb : [N, D]
    return: indices[B, K], scores[B, K]  （按余弦相似度降序）
    """
    Q = queries   # [B, D]
    X = db_emb    # [N, D]
    B, D = Q.shape
    N = X.shape[0]
    k = int(min(k, N))

    sims = Q @ X.T            # [B, N]
    if k == N:
        order = np.argsort(-sims, axis=1)
        topk_idx = order[:, :k]
        topk_scores = np.take_along_axis(sims, topk_idx, axis=1)
    else:
        part = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]         # [B, k]
        part_scores = np.take_along_axis(sims, part, axis=1)          # [B, k]
        order_in_part = np.argsort(-part_scores, axis=1)              # [B, k]
        topk_idx = np.take_along_axis(part, order_in_part, axis=1)    # [B, k]
        topk_scores = np.take_along_axis(part_scores, order_in_part, axis=1)
    return topk_idx.astype(np.int64), topk_scores.astype(np.float32)

# _topk_sim_batch
# _stratified_k_sim_batch
def _stratified_k_sim_batch(queries, db_emb, k, bins=(0.0, 0.3, 0.7, 1.0), quota=(2, 3, 1), seed=None):
    """
    将相似度降序列表按分位切成多段，每段抽 quota 个（不足则尽量抽），合计 ~k 个。
    bins: 分位边界（含左不含右，最后一段含右）
    quota: 每段配额之和建议≈k
    """
    assert len(bins) >= 2 and len(quota) == len(bins) - 1
    Q = _ensure_l2(queries); X = _ensure_l2(db_emb)
    B, N = Q.shape[0], X.shape[0]
    k = int(min(k, N))
    sims = Q @ X.T
    order = np.argsort(-sims, axis=1)

    rng = np.random.default_rng(seed)
    out_idx = np.empty((B, k), dtype=np.int64)
    out_scores = np.empty((B, k), dtype=np.float32)

    for i in range(B):
        picks = []
        for b in range(len(quota)):
            start = int(N * bins[b])
            end   = int(N * bins[b+1])
            end = max(end, start + 1)
            band = order[i, start:end]
            q = quota[b]
            if band.size <= q:
                chosen = band
            else:
                chosen = rng.choice(band, size=q, replace=False)
            picks.append(chosen)

        chosen = np.concatenate(picks) if picks else np.array([], dtype=np.int64)
        # 若总数不足 k，从未使用部分按相似度降序补齐
        if chosen.size < k:
            mask = np.ones(N, dtype=bool)
            mask[chosen] = False
            remainder = order[i, mask]
            chosen = np.concatenate([chosen, remainder[:(k - chosen.size)]])
        elif chosen.size > k:
            # 若超了，就按分数裁掉尾部
            s = sims[i, chosen]
            o = np.argsort(-s)
            chosen = chosen[o[:k]]

        s = sims[i, chosen]
        o = np.argsort(-s)
        out_idx[i] = chosen[o]
        out_scores[i] = s[o].astype(np.float32)

    return out_idx.astype(np.int64), out_scores.astype(np.float32)
    # return out_idx, out_scores


# def _topk_sim_batch(queries, db_emb, k, lower=0.1, upper=0.3, random_pick=True):
def _topk_sim_batch(queries, db_emb, k, lower=0.3, upper=0.7, random_pick=True, top=False, down=False):
    """
    在每个查询的相似度降序序列中，截取 [lower, upper) 这个相对区间（如 30%-70%）
    再从该区间选出 k 个（可随机选，或带内取高分）。
    """
    assert 0.0 <= lower < upper <= 1.0
    Q = (queries)   # [B, D]
    
    X = (db_emb)
    if isinstance(X, torch.Tensor):
        X = (db_emb).cpu().numpy()    # [N, D]
          
    B, D = Q.shape
    N = X.shape[0]
    k = int(min(k, N))

    sims = Q @ X.T            # [B, N]
    order = np.argsort(-sims, axis=1)  # 各行从高到低排序索引

    start = (N * lower).astype(int) if isinstance(lower, np.ndarray) else int(N * lower)
    end   = (N * upper).astype(int) if isinstance(upper, np.ndarray) else int(N * upper)
    end = max(end, start + 1)  # 至少有1个


    topk_idx = np.empty((B, k), dtype=np.int64)
    topk_scores = np.empty((B, k), dtype=np.float32)

    for i in range(B):
        band_idx_sorted = order[i, start:end]  # 中间带的索引（已按相似度降序）
        if band_idx_sorted.size == 0:
            # 退化：若区间太窄导致空集，回退到普通 top-k
            topk_idx[i] = order[i, :k]
            topk_scores[i] = sims[i, topk_idx[i]]
            continue
        
        if random_pick:
            # 在带内无放回随机采样 k 个
            if band_idx_sorted.size <= k:
                chosen = band_idx_sorted
            else:
                chosen = np.random.choice(band_idx_sorted, size=k, replace=False)

            # 为了输出有序，可按分数再排一下
            chosen_scores = sims[i, chosen]
            sort_in = np.argsort(-chosen_scores)
            chosen = chosen[sort_in]
            chosen_scores = chosen_scores[sort_in]
        elif down:
            chosen = band_idx_sorted[-k:]
            chosen_scores = sims[i, chosen]
        elif top:
            # 不随机：直接取“带内的前 k 个”（带内仍按相似度降序）
            chosen = band_idx_sorted[:k]
            chosen_scores = sims[i, chosen]

        # 若带内不足 k 个，可从左右边界再补齐（可选）
        if chosen.size < k:
            # 从带外（优先靠近带的两侧）补齐
            left = order[i, :start]
            right = order[i, end:]
            # 交替从左右拿，直到补够
            fill = []
            li, ri = 0, 0
            need = k - chosen.size
            while need > 0 and (li < left.size or ri < right.size):
                if li < left.size:
                    fill.append(left[li]); li += 1; need -= 1
                    if need == 0: break
                if ri < right.size:
                    fill.append(right[ri]); ri += 1; need -= 1
            if fill:
                fill = np.array(fill, dtype=np.int64)
                chosen = np.concatenate([chosen, fill[:k - chosen.size]])
                chosen_scores = sims[i, chosen]
            # 最终再按分数降序
            o = np.argsort(-chosen_scores)
            chosen = chosen[o]; chosen_scores = chosen_scores[o]

        topk_idx[i] = chosen.astype(np.int64)
        topk_scores[i] = chosen_scores.astype(np.float32)

    # return topk_idx, topk_scores
    return topk_idx.astype(np.int64), topk_scores.astype(np.float32)

def get_more_data_filter(
    retri_db,
    o_a, o_v,
    score_a_data, score_v_data,
    y_query_a=None, y_query_v=None,   # [B] 或 [B,1]，支持 batch
    use_l2=True,
    lower_a=0.3 , upper_a = 0.7,
    lower_v=0.3 , upper_v = 0.7,
    random_pick_a=True , top_a=False , down_a=False,
    random_pick_v=True , top_v=False , down_v=False
):
    """
    返回:
    {
      "audio": {"indices": [B,K], "scores": [B,K], "labels": [B,K] or None},
      "video": {"indices": [B,K], "scores": [B,K], "labels": [B,K] or None},
    }
    """

    if "logits_raw_train_audio_logits" in retri_db:
        logits_a = retri_db["logits_raw_train_audio_logits"]
    else:
        raise KeyError("Audio embeddings not found.")

    if "logits_raw_train_audio_video_e_logits" in retri_db:
        logits_v = retri_db["logits_raw_train_audio_video_e_logits"]
    else:
        raise KeyError("Video embeddings not found.")

    labels_a = retri_db.get("labels_train_audio_logits", None)
    labels_v = retri_db.get("labels_train_audio_video_e_logits", None)


    # ----- 将查询展平为 [B, D] 并拿到 B -----
    Qa = _as_2d(o_a)   # [B,D]
    Qv = _as_2d(o_v)   # [B,D]
    B = Qa.shape[0]
    K_a = int(score_a_data)
    K_v = int(score_v_data)

    # ----- 如果没有标签过滤，直接全库检索（保持与原逻辑一致） -----
    if (labels_a is None or y_query_a is None) and (labels_v is None or y_query_v is None):
        Ia, Sa = _topk_sim_batch(Qa, logits_a, K_a, lower_a, upper_a, random_pick=random_pick_a, top=top_a, down=down_a)  # [B,K]
        Iv, Sv = _topk_sim_batch(Qv, logits_v, K_v, lower_v, upper_v, random_pick=random_pick_v, top=top_v, down=down_v)
        La = (labels_a[Ia] if labels_a is not None else None)
        Lv = (labels_v[Iv] if labels_v is not None else None)
        return {"audio": {"indices": Ia, "scores": Sa, "labels": La},
                "video": {"indices": Iv, "scores": Sv, "labels": Lv}}
    
    # ====== 关键修改：按“查询标签”分组做检索 ======

    def _ensure_1d_cpu_np(x):
        if x is None:
            return None
        # 兼容 torch.Tensor / numpy / list
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.asarray(x).reshape(-1)        # -> [B]
        return x

    yqa = _ensure_1d_cpu_np(y_query_a)   # [B] or None
    yqv = _ensure_1d_cpu_np(y_query_v)   # [B] or None

    # 结果容器
    Ia_full = np.zeros((B, K_a), dtype=np.int64)
    Sa_full = np.zeros((B, K_a), dtype=np.float32)
    Iv_full = np.zeros((B, K_v), dtype=np.int64)
    Sv_full = np.zeros((B, K_v), dtype=np.float32)

    # ---- 音频：按标签分组 ----
    if labels_a is not None and yqa is not None:
        labels_a_np = np.asarray(labels_a)
        uniq_a = np.unique(yqa)  # 只跑这些类
        for c in uniq_a:
            # 库内该类的候选索引
            sel_idx = np.where(labels_a_np == c)[0]     # [M_c]
            if sel_idx.size == 0:
                # 该类在库里没有样本：可以退回全库、或返回空；这里退回全库以避免全空
                emb_sub = logits_a
                map_back = None
            else:
                emb_sub = logits_a[sel_idx]
                map_back = sel_idx

            # 这一类的查询行
            rows = np.where(yqa == c)[0]                # [R]
            Qa_c = Qa[rows]                              # [R,D]


            I_sub, S_sub = _topk_sim_batch(Qa_c, emb_sub, K_a, lower_a, upper_a, random_pick=random_pick_a, top=top_a, down=down_a)   # [R,K]
            if map_back is not None:
                I_full = map_back[I_sub]
            else:
                I_full = I_sub

            Ia_full[rows] = I_full
            Sa_full[rows] = S_sub
    else:
        # 没有标签过滤：全库
        Ia_full, Sa_full = _topk_sim_batch(Qa, logits_a, K_a, lower_a, upper_a, random_pick=random_pick_a, top=top_a, down=down_a)

    # ---- 视频：按标签分组 ----
    if labels_v is not None and yqv is not None:
        labels_v_np = np.asarray(labels_v)
        uniq_v = np.unique(yqv)
        for c in uniq_v:
            sel_idx = np.where(labels_v_np == c)[0]
            if sel_idx.size == 0:
                emb_sub = logits_v
                map_back = None
            else:
                emb_sub = logits_v[sel_idx]
                map_back = sel_idx

            rows = np.where(yqv == c)[0]
            Qv_c = Qv[rows]

            I_sub, S_sub = _topk_sim_batch(Qv_c, emb_sub, K_v, lower_v, upper_v, random_pick=random_pick_v, top=top_v, down=down_v)
            if map_back is not None:
                I_full = map_back[I_sub]
            else:
                I_full = I_sub

            Iv_full[rows] = I_full
            Sv_full[rows] = S_sub
    else:
        Iv_full, Sv_full = _topk_sim_batch(Qv, logits_v, K_v, lower_v, upper_v, random_pick=random_pick_v, top=top_v, down=down_v)

    # 标签收集
    La = (labels_a[Ia_full] if labels_a is not None else None)
    Lv = (labels_v[Iv_full] if labels_v is not None else None)

    return {
        "audio": {"indices": Ia_full, "scores": Sa_full, "labels": La},
        "video": {"indices": Iv_full, "scores": Sv_full, "labels": Lv},
    }

def shrink_pairs(A, V, pair_cap, min_each=1):
    """
    把现有的 A、V 等比缩放到不超过 pair_cap，且尽量保持 A:V 比例。
    """
    if A <= 0:
        A = 1
    elif V <= 0:
        V = 1
    prod = A * V
    if prod <= pair_cap:
        return A, V

    # 20 12   32 
    # 等比缩放系数（几何缩放）
    s = math.sqrt(pair_cap / prod)
    A2 = max(min_each, int(math.floor(A * s)))
    V2 = max(min_each, int(math.floor(V * s)))

    # 可能因为取整造成 A2*V2 < pair_cap，尝试小幅回填
    # 贪心把空间分给“边际增益更大”的那个维度
    while (A2 + 1) * V2 <= pair_cap and A2 + 1 <= A:
        A2 += 1
    while A2 * (V2 + 1) <= pair_cap and V2 + 1 <= V:
        V2 += 1

    # 若仍然超（极少见于边界），再微调
    while A2 * V2 > pair_cap:
        if A2 >= V2 and A2 > min_each:
            A2 -= 1
        elif V2 > min_each:
            V2 -= 1
        else:
            break
    return A2, V2


def local_prototype_pull_loss(
    query_emb: torch.Tensor,          # [B, D] 当前样本的embedding（需要回传梯度）
    query_labels: torch.Tensor,        # [B]
    retrieved_emb: torch.Tensor,       # [B, K, D] 从库检索到的emb（常量/不回传）
    retrieved_labels: torch.Tensor,    # [B, K]
    tau: float = 0.07,
    use_weighted_proto: bool = True,   # True: 相似度加权中心（更稳）；False: 直接均值中心
    min_pos: int = 1,                  # 至少需要几个同类近邻才计算该样本loss
    reduction: str = "mean",
):
    """
    显式聚类：把 query_emb 拉向从检索同类样本算出的 prototype。
    返回: loss (scalar), valid_mask ([B] bool)
    """
    assert query_emb.dim() == 2
    B, D = query_emb.shape
    assert retrieved_emb.dim() == 3 and retrieved_emb.shape[0] == B and retrieved_emb.shape[2] == D
    assert retrieved_labels.shape[:2] == retrieved_emb.shape[:2]

    # normalize 用 cosine 距离更稳定
    q = F.normalize(query_emb, dim=-1)                    # [B, D]
    r = F.normalize(retrieved_emb, dim=-1)                # [B, K, D]

    # 同类mask：retrieved_labels == query_labels
    pos_mask = (retrieved_labels == query_labels[:, None])  # [B, K] bool
    pos_cnt = pos_mask.sum(dim=1)                           # [B]
    valid = pos_cnt >= min_pos                              # [B] bool

    # 没有同类检索到时：避免 NaN（这些样本 loss 设为 0 并在 reduction 时跳过）
    # 先准备一个安全mask
    safe_mask = pos_mask & valid[:, None]                  # [B, K]

    if use_weighted_proto:
        # 相似度加权 prototype：w = softmax(sim(q, r)/tau) 但只在同类上归一化
        # sim: [B, K]
        sim = (q[:, None, :] * r).sum(dim=-1)  # cosine sim
        sim = sim / tau

        # 把非同类设为 -inf，使 softmax 只在同类上分配权重
        sim_masked = sim.masked_fill(~safe_mask, float("-inf"))
        w = torch.softmax(sim_masked, dim=1)               # [B, K]；无同类行会变成 nan
        w = torch.nan_to_num(w, nan=0.0)                   # 无同类行权重全0

        proto = torch.einsum("bk,bkd->bd", w, r)            # [B, D]
    else:
        # 直接均值 prototype（只平均同类）
        masked_r = r * safe_mask[..., None].float()         # [B, K, D]
        denom = safe_mask.sum(dim=1).clamp_min(1).float()   # [B]
        proto = masked_r.sum(dim=1) / denom[:, None]        # [B, D]

    proto = F.normalize(proto, dim=-1).detach()             # stopgrad: prototype不回传

    # pull loss：1 - cosine(q, proto)
    cos = (q * proto).sum(dim=-1)                           # [B]
    loss_per = 1.0 - cos                                    # [B]
    loss_per = loss_per * valid.float()                     # invalid样本置0

    if reduction == "mean":
        denom = valid.float().sum().clamp_min(1.0)
        return loss_per.sum() / denom, valid
    elif reduction == "sum":
        return loss_per.sum(), valid
    else:
        return loss_per, valid
    
## merge_alpha set is 0.4
def train_audio_video(epoch, train_loader, model, optimizer, logger, count_add_a, 
                      count_add_v, gs_plugin=None, merge_alpha=0.5, retri_db = None, 
                      epoch_thre=10, thre_sample=10, thre_sample_min=2, warmup_epochs=10, 
                      tmp = 0.5, lower_a = 0.1 , upper_a = 0.1, lower_v = 0.1 , upper_v = 0.1, 
                      logits_ratio = 0, merge_alpha_new_a = 0.5, merge_alpha_new_v = 0.5, min_alpha = 1.0,
                      random_pick_a=True , top_a=False , down_a=False,
                      random_pick_v=True , top_v=False , down_v=False, new_encoder = 10, all_updata_epoch = 10):
    
    data_more_o_a_sum = 0
    data_more_o_v_sum = 0
    model.train()
    
    global s_a
    global s_v
    # ----- RECORD LOSS AND ACC -----
    tl = Averager()
    t_new = Averager()
    ta = Averager()
    tv = Averager()
    len_dataloader = len(train_loader)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    ## 105 
    # print(f"len(train_loader) : {len(train_loader)}")

    ## all before 2 is this
    # temp_step = new_encoder + (epoch % 10)

    for step, (spectrogram, image, y) in enumerate(tqdm(train_loader, desc="Traning")):
                                      
        score_v = 0.0   
        score_a = 0.0

        image = image.float().cuda()
        device = image.device
        y = y.cuda()
        bs = y.shape[0]
        class_num = y.shape[1]
        
        spectrogram = spectrogram.unsqueeze(1).float().cuda()
        optimizer.zero_grad()
        
        o_b, o_a, o_v, a_f, v_f = model(spectrogram, image)
        out_v = o_v / tmp
        out_a = o_a / tmp
        
        loss_a = criterion(o_a, y).mean()
        loss_v = criterion(o_v, y).mean()
        
        # loss_fusion = criterion( 0.5 * o_v +  0.5 * o_a, y).mean()
        
        ## TODO:11-11
        ## 如果乘的是 0.5 则训练更温和
        ## 如果乘的是 1.0 则是正常训练

        loss_new = torch.tensor(0)
        
        pred_v = torch.argmax(out_v, dim=1)
        pred_a = torch.argmax(out_a, dim=1)
        true = torch.argmax(y, dim=1)
        score_v = (pred_v == true).sum().item()
        score_a = (pred_a == true).sum().item()

        ratio_imbalance = score_a / score_v
        writer.add_scalars('ratio_imbalance', {
            'ratio_imbalance': ratio_imbalance,
        },  epoch * len(train_loader) + step)
        
        # if (epoch < epoch_thre) or (epoch % all_updata_epoch != 0):
        if (epoch < epoch_thre):

            loss_fusion = criterion( logits_ratio * o_v + logits_ratio * o_a, y).mean()
            loss_new = torch.tensor(0)
            loss = loss_fusion + loss_a + loss_v
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ratio_a = 0        
        
        elif epoch >= epoch_thre:
            ratio_s_a = score_v / (score_v + score_a) 
            ratio_s_v = score_a / (score_v + score_a)
            score_a_data = ratio_s_a * thre_sample
            score_v_data = ratio_s_v * thre_sample

            score_a_data = int(score_a_data  + 0.5)
            score_v_data = int(score_v_data  + 0.5)

            data_more_o_a_sum += score_a_data
            data_more_o_v_sum += score_v_data

            writer.add_scalars('Scores', {
                'audio': score_a_data,
                'video': score_v_data,
            },  epoch * len(train_loader) + step)

            y_idx = y.argmax(dim=1)
            result = get_more_data_filter(retri_db, o_a, o_v, score_a_data, score_v_data, y_query_a=y_idx , y_query_v=y_idx, 
                                          lower_a=lower_a , upper_a = upper_a, lower_v=lower_v , upper_v = upper_v, 
                                          random_pick_a=random_pick_a , top_a=top_a , down_a=down_a,
                                          random_pick_v=random_pick_v , top_v=top_v , down_v=down_v)
            
            audio_idx = result["audio"]["indices"]
            video_idx = result["video"]["indices"]

            # print("OLD")
            audio_embs = retri_db["embeddings_raw_train_audio_logits"][audio_idx]
            video_embs = retri_db["embeddings_raw_train_audio_video_e_logits"][video_idx]
            
            audio_embs = torch.from_numpy(audio_embs).to(device=image.device, dtype=torch.float32)
            video_embs = torch.from_numpy(video_embs).to(device=image.device, dtype=torch.float32)
            
            audio_labels = retri_db["labels_train_audio_logits"][audio_idx]
            video_labels = retri_db["labels_train_audio_video_e_logits"][video_idx]

            audio_labels = torch.from_numpy(audio_labels).to(device=image.device)
            video_labels = torch.from_numpy(video_labels).to(device=image.device)
            
            # audio_embs = audio_embs.reshape(-1, audio_embs.shape[-1])      # [B*K, D]
            # audio_labels = audio_labels.reshape(-1)                          # [B*K]
            
            # video_embs = video_embs.reshape(-1, video_embs.shape[-1])       # [B*K2, D]
            # video_labels = video_labels.reshape(-1)                          # [B*K2]

            o_b, out_a, out_v = model.clssifier(audio_embs, video_embs)

            # audio_labels = retri_db["labels_train_audio_logits"][audio_idx]
            # video_labels = retri_db["labels_train_audio_video_e_logits"][video_idx]

            # audio_labels = torch.from_numpy(audio_labels).to(device=image.device)
            # video_labels = torch.from_numpy(video_labels).to(device=image.device)

            # audio_labels = audio_labels.reshape(-1)                          # [B*K]
            # video_labels = video_labels.reshape(-1)                          # [B*K2]
            
            num_classes = y.shape[1]
            video_labels = F.one_hot(video_labels.long(), num_classes=num_classes).float()
            audio_labels = F.one_hot(audio_labels.long(), num_classes=num_classes).float()
            
            audio_labels_te = audio_labels.argmax(dim=-1)
            video_labels_te = video_labels.argmax(dim=-1)

            z_a_cur = a_f  # [B, D] 你当前样本的音频/文本分支embedding
            z_v_cur = v_f  # [B, D] 你当前样本的视频/图像分支embedding
            y_cur   = y.argmax(dim=-1)   # [B]

            # query_emb: torch.Tensor,          # [B, D] 当前样本的embedding（需要回传梯度）
            # query_labels: torch.Tensor,        # [B]
            # retrieved_emb: torch.Tensor,       # [B, K, D] 从库检索到的emb（常量/不回传）
            # retrieved_labels: torch.Tensor,    # [B, K]

            loss_pull_a, valid_a = local_prototype_pull_loss(
                query_emb=z_a_cur,
                query_labels=y_cur,
                retrieved_emb=audio_embs,         # [B, K, D]
                retrieved_labels=audio_labels_te,    # [B, K]
                use_weighted_proto=False,
                tau=0.07,
            )
            
            loss_pull_v, valid_v = local_prototype_pull_loss(
                query_emb=z_v_cur,
                query_labels=y_cur,
                retrieved_emb=video_embs,         # [B, K2, D]
                retrieved_labels=video_labels_te,    # [B, K2]
                use_weighted_proto=False,
                tau=0.07,
            )

            loss_pull = loss_pull_a + loss_pull_v
            loss_pull = loss_pull * 0.1

            C = class_num
            B = bs
            A = score_a_data
            V = score_v_data
            
            
            ## TODO:really doubt
            # loss_fusion = criterion( logits_ratio * o_v + logits_ratio * o_a, y).mean() / 23 / 6
            loss_fusion = criterion( logits_ratio * o_v + logits_ratio * o_a, y).mean()
            
            out_a = out_a / tmp
            # print("out_a.shape ", out_a.shape)
            out_v = out_v / tmp
            # print("1111")


            out_a = out_a.reshape(bs, score_a_data, class_num)
            out_v = out_v.reshape(bs, score_v_data, class_num)
            o_v_expand = o_v.unsqueeze(dim=1)
            o_a_expand = o_a.unsqueeze(dim=1)
            y_expand = y.unsqueeze(dim=1)



            out_a_all = torch.cat([out_a, o_a_expand], dim=1)
            out_v_all = torch.cat([out_v, o_v_expand], dim=1)
            
            add_num = 1
            video_labels = video_labels.reshape(bs, score_v_data, class_num )
            video_labels_all = torch.cat([video_labels, y_expand], dim=1)
            audio_labels = audio_labels.reshape(bs, score_a_data, class_num )
            audio_labels_all = torch.cat([audio_labels, y_expand], dim=1)

            video_labels_all = video_labels_all.unsqueeze(1).expand(B, A + add_num, V + add_num, C)
            audio_labels_all = audio_labels_all.unsqueeze(2).expand(B, A + add_num, V + add_num, C)
            
            all_labels = video_labels_all * merge_alpha + audio_labels_all * (1 - merge_alpha)     # [B,2,4,C]
            # all_labels = all_labels.view(B * (A + add_num) * (V + add_num), C)                       # [B*8, C]
            
            
            B, A, C = out_a.shape
            _, V, _ = out_v.shape
            a = out_a_all.unsqueeze(2).expand(B, (A + add_num), 1, C)
            # [B,1,4,C] -> [B,2,4,C]
            v = out_v_all.unsqueeze(1).expand(B, 1, (V + add_num) , C)
            
            # fusion_out_new = a * merge_alpha + v * (1.0 - merge_alpha)  # [B,2,4,C]
            fusion_out_new = a * merge_alpha_new_a + v * merge_alpha_new_v  # [B,2,4,C]
            # fusion_out_new = a + v  # [B,2,4,C]            
            fusion_out_new = fusion_out_new.reshape(B, (A + add_num), (V + add_num), C)
            
            loss_new = criterion(fusion_out_new, all_labels).mean()
            
            # alpha = min(min_alpha, epoch / warmup_epochs)
            loss = loss_new + loss_a + loss_v + loss_pull
            
            # loss = loss_new 
            # loss = loss_new * alpha
            # loss = loss_fusion
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            # gs_plugin.exp_count += 1

            ratio_a = score_a / score_v    

        tl.add(loss_fusion.item())
        t_new.add(loss_new.item())
        ta.add(loss_a.item())
        tv.add(loss_v.item())

        ## clear gradient   
        for n, p in model.named_parameters():
            if p.grad != None:
                del p.grad

        if step % cfg['print_inteval'] == 0:
            logger.info((
                'Epoch:{epoch}, Trainnig Loss:{train_loss:.3f}, Training Loss_a:{loss_a:.3f}, Training Loss_v:{loss_v:.3f}, Training Loss_new:{loss_new:.3f},  Training loss_fusion:{loss_fusion:.3f}'
                ).format(epoch=epoch, train_loss=loss.item(), loss_a=loss_a.item(), loss_v=loss_v.item(), loss_new=loss_new.item(), loss_fusion=loss_fusion.item()))
    
    print(f"data_more_o_a_sum: {data_more_o_a_sum} ///  data_more_o_v_sum: {data_more_o_v_sum}")
    
    loss_ave = tl.item()
    loss_new_ave = t_new.item()
    loss_a_ave = ta.item()
    loss_v_ave = tv.item()

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}:  Average Training_new Loss:{loss_new_ave:.3f}, Average Training Loss:{loss_ave:.3f}, Average Training Loss_a:{loss_a_ave:.2f}, Average Training Loss_v:{loss_v_ave:.2f}').format(
        epoch=epoch, loss_new_ave=loss_new_ave, loss_ave=loss_ave, loss_a_ave=loss_a_ave, loss_v_ave=loss_v_ave))
    # 记录平均损失到 TensorBoard

    writer.add_scalar('Loss/loss_ave', loss_ave, epoch)
    writer.add_scalar('Loss/loss_new_ave', loss_new_ave, epoch)
    writer.add_scalar('Loss/loss_a_ave', loss_a_ave, epoch)
    writer.add_scalar('Loss/loss_v_ave', loss_v_ave, epoch)
    
    return model, ratio_a, score_a, score_v, data_more_o_a_sum, data_more_o_v_sum

def val(epoch, val_loader, model, logger, merge_alpha=0.5, writer=None):

    model.eval()
    pred_list = []
    pred_list_three = []
    pred_list_four = []

    pred_list_a = []
    pred_list_v = []
    pred_list_logits_all = []

    label_list = []

    soft_pred = []
    soft_pred_four = []
    soft_pred_three = []

    soft_pred_a = []
    soft_pred_v = []
    soft_pred_logits = []

    one_hot_label = []

    score_a = 0.0
    score_v = 0.0
    
    with torch.no_grad():
        for step, (spectrogram, image, y) in enumerate(tqdm(val_loader, desc="val")):

            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            image = image.cuda()
            y = y.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()
 
            o_b ,out_a, out_v, _ ,_ = model(spectrogram, image)

            # out = merge_alpha * out_a + (1 - merge_alpha) * out_v
            out = out_a * 2 + out_v * 2
            out_four = out_a * 0.4 + out_v * 0.6
            out_three = out_a * 0.3 + out_v * 0.7

            # criterion(out_a, y)

            ## model predict answer's max probability
            soft_pred_a = soft_pred_a + (F.softmax(out_a, dim=1)).tolist()
            soft_pred_v = soft_pred_v + (F.softmax(out_v, dim=1)).tolist()
            soft_pred = soft_pred + (F.softmax(out, dim=1)).tolist()

            soft_pred_four = soft_pred_four + (F.softmax(out_four, dim=1)).tolist()
            soft_pred_three = soft_pred_three + (F.softmax(out_three, dim=1)).tolist()

            soft_pred_logits = soft_pred_logits + ((F.softmax(out_a, dim=1) + F.softmax(out_v, dim=1)) / 2).tolist()


            pred = F.softmax(out, dim=1).argmax(dim=1)
            pred_four = F.softmax(out_four, dim=1).argmax(dim=1)
            pred_three = F.softmax(out_three, dim=1).argmax(dim=1)

            pred_a = (F.softmax(out_a, dim=1)).argmax(dim=1)
            pred_v = (F.softmax(out_v, dim=1)).argmax(dim=1)
            pred_list_logits = ((F.softmax(out_a, dim=1) + F.softmax(out_v, dim=1)) / 2).argmax(dim=1)
            
            pred_list = pred_list + pred.tolist()

            pred_list_three = pred_list_three + pred_three.tolist()
            pred_list_four = pred_list_four + pred_four.tolist()



            pred_list_a = pred_list_a + pred_a.tolist()
            pred_list_v = pred_list_v + pred_v.tolist()
            pred_list_logits_all = pred_list_logits_all + pred_list_logits.tolist()


        f1 = f1_score(label_list, pred_list, average='macro')
        f1_3 = f1_score(label_list, pred_list_three, average='macro')
        f1_4 = f1_score(label_list, pred_list_four, average='macro')
        f1_a = f1_score(label_list, pred_list_a, average='macro')
        f1_v = f1_score(label_list, pred_list_v, average='macro')
        f1_logits = f1_score(label_list, pred_list_logits_all, average='macro')

        correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
        correct_3 = sum(1 for x, y in zip(label_list, pred_list_three) if x == y)
        correct_4 = sum(1 for x, y in zip(label_list, pred_list_four) if x == y)

        correct_a = sum(1 for x, y in zip(label_list, pred_list_a) if x == y)
        correct_v = sum(1 for x, y in zip(label_list, pred_list_v) if x == y)
        correct_logits = sum(1 for x, y in zip(label_list, pred_list_logits_all) if x == y)

        acc = correct / len(label_list)

        acc_3 = correct_3 / len(label_list)
        acc_4 = correct_4 / len(label_list)
        
        acc_a = correct_a / len(label_list)
        acc_v = correct_v / len(label_list)
        acc_logits = correct_logits / len(label_list)

        mAP = compute_mAP(torch.Tensor(soft_pred), torch.Tensor(one_hot_label))
        mAP_3 = compute_mAP(torch.Tensor(soft_pred_three), torch.Tensor(one_hot_label))
        mAP_4 = compute_mAP(torch.Tensor(soft_pred_four), torch.Tensor(one_hot_label))
 
        mAP_a = compute_mAP(torch.Tensor(soft_pred_a), torch.Tensor(one_hot_label))
        mAP_v = compute_mAP(torch.Tensor(soft_pred_v), torch.Tensor(one_hot_label))
        mAP_logits = compute_mAP(torch.Tensor(soft_pred_logits), torch.Tensor(one_hot_label))


    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: f1_3:{f1_3:.4f},acc_3:{acc_3:.4f},mAP_3:{mAP_3:.4f}, \
                 f1_4:{f1_4:.4f},acc_4:{acc_4:.4f},mAP_4:{mAP_4:.4f}, \
                 f1_logits:{f1_logits:.4f},acc_logits:{acc_logits:.4f},mAP_logits:{mAP_logits:.4f}, \
                 f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},f1_a:{f1_a:.4f},acc_a:{acc_a:.4f}, mAP_a:{mAP_a:.4f}, \
                 f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}')
                    .format(epoch=epoch, 
                    f1_3=f1_3, acc_3=acc_3, mAP_3=mAP_3,
                    f1_4=f1_4, acc_4=acc_4, mAP_4=mAP_4,
                    f1_logits=f1_logits, acc_logits=acc_logits, mAP_logits=mAP_logits,
                    f1=f1, acc=acc, mAP=mAP,
                    f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
                    f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v))
    

    writer.add_scalar('f1/ALL', f1, epoch)
    writer.add_scalar('acc/ALL', acc, epoch)
    writer.add_scalar('mAP/ALL', mAP, epoch)

    writer.add_scalar('f1_logits/ALL', f1_logits, epoch)
    writer.add_scalar('acc_logits/ALL', acc_logits, epoch)
    writer.add_scalar('mAP_logits/ALL', mAP_logits, epoch)
    
    writer.add_scalar('f1_a/audio', f1_a, epoch)
    writer.add_scalar('mAP_a/audio', mAP_a, epoch)
    writer.add_scalar('acc_a/audio', acc_a, epoch)


    writer.add_scalar('f1_v/image', f1_v, epoch)
    writer.add_scalar('mAP_v/image', mAP_v, epoch)
    writer.add_scalar('acc_v/image', acc_v, epoch)

    writer.add_scalar('f1_3/image', f1_3, epoch)
    writer.add_scalar('mAP_3/image', mAP_3, epoch)
    writer.add_scalar('acc_3/image', acc_3, epoch)

    writer.add_scalar('f1_4/image', f1_4, epoch)
    writer.add_scalar('mAP_4/image', mAP_4, epoch)
    writer.add_scalar('acc_4/image', acc_4, epoch)

    log_metrics(epoch, acc_logits, mAP_logits, f1_logits , acc, acc_a, acc_v, f1, f1_a, f1_v, mAP, mAP_a, mAP_v)

    logger.info(('s_a:{s_a:.3f}, s_v:{s_v:.3f}').format(s_a=s_a, s_v=s_v))
    return acc, score_a, score_v


# /data/lxe_2024/2343_old_file/ori_unimodel/train_audio_resnet18_resnet18_0.6075268817204301_91_best_model.pth
# /data/lxe_2024/2343_old_file/ori_unimodel/train_video_resnet18_resnet18_0.40860215053763443_98_best_model.pth

def retri_produce_embed_sim_db(model, train_loader):
    logits_a_list = []
    logits_v_list = []

    feats_a_list = []
    feats_v_list = []
    
    model.eval()
    with torch.no_grad():
        for step, (spectrogram, image, y) in enumerate(tqdm(train_loader, desc="DB extract feature")):
            image = image.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()
            o_b ,out_a, out_v, f_a , f_v = model(spectrogram, image)
            out_a = out_a.detach().float().cpu()
            out_v = out_v.detach().float().cpu()

            f_a = f_a.detach().float().cpu()
            f_v = f_v.detach().float().cpu()

            logits_a_list.append(out_a)
            logits_v_list.append(out_v)

            feats_a_list.append(f_a)
            feats_v_list.append(f_v)
        
        logits_a = torch.cat(logits_a_list, dim=0).numpy().astype('float32')
        logits_v = torch.cat(logits_v_list, dim=0).numpy().astype('float32')
        feats_a = torch.cat(feats_a_list, dim=0).numpy().astype('float32')
        feats_v = torch.cat(feats_v_list, dim=0).numpy().astype('float32')

    return logits_a, logits_v, feats_a, feats_v

def read_db(paths):
    retri_db = {}
    for path in paths:
        key = path.split('/')[-1].replace('.npy', '')  # 例如 "labels_train_audio"
        retri_db[key] = np.load(path)
    return retri_db

def log_metrics(epoch,  acc_logits, mAP_logits, f1_logits ,acc, acc_a, acc_v, f1, f1_a, f1_v, mAP, mAP_a, mAP_v):
    """
    将各项指标记录到 txt 文件中
    """
    with open(log_file_path, "a") as f:
        f.write(
            f"Epoch {epoch:03d} | "
            f"acc_logits: {acc_logits:.4f} | mAP_logits: {mAP_logits:.4f} | f1_logits: {f1_logits:.4f} | "
            f"Acc: {acc:.4f} | Acc_A: {acc_a:.4f} | Acc_V: {acc_v:.4f} | "
            f"F1: {f1:.4f} | F1_A: {f1_a:.4f} | F1_V: {f1_v:.4f} | "
            f"mAP: {mAP:.4f} | mAP_A: {mAP_a:.4f} | mAP_V: {mAP_v:.4f}\n"
        )


@torch.no_grad()
def _ema_update(old_t, new_t, decay: float):
    # old 不存在时，直接等于 new
    if old_t is None:
        return new_t
    # 形状不一致时，回退为直接覆盖（或者你也可以 assert）
    if tuple(old_t.shape) != tuple(new_t.shape):
        return new_t
    return decay * old_t + (1.0 - decay) * new_t

def make_new_retri_db(model, train_loader, retri_db, epoch, decay):

    logits_a, logits_v, feats_a, feats_v = retri_produce_embed_sim_db(model, train_loader)
    logits_a_t = torch.tensor(logits_a, device='cuda')
    logits_a_t = logits_a_t.cpu().numpy()
    logits_v_t = torch.tensor(logits_v, device='cuda')
    logits_v_t = logits_v_t.cpu().numpy()

    old_logits_a = retri_db.get("logits_raw_train_audio_logits", None)
    old_logits_v = retri_db.get("logits_raw_train_audio_video_e_logits", None)
    old_feats_a  = retri_db.get("embeddings_raw_train_audio_logits", None)
    old_feats_v  = retri_db.get("embeddings_raw_train_audio_video_e_logits", None)

    logits_a_ema = _ema_update(old_logits_a, logits_a_t, decay)
    logits_v_ema = _ema_update(old_logits_v, logits_v_t, decay)
    feats_a_ema  = _ema_update(old_feats_a,  feats_a,  decay)
    feats_v_ema  = _ema_update(old_feats_v,  feats_v,  decay)
    

    retri_db["embeddings_raw_train_audio_logits"] = feats_a_ema
    retri_db["embeddings_raw_train_audio_video_e_logits"] = feats_v_ema
    retri_db["logits_raw_train_audio_video_e_logits"] = logits_v_ema
    retri_db["logits_raw_train_audio_logits"] = logits_a_ema
    # retri_db["sim_a"] = sim_a
    # retri_db["sim_v"] = sim_v
    
    return retri_db

# 在主函数中初始化 SummaryWriter
# 当前日期和时间（格式：YYYYMMDD_HHMMSS）
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
tips = "NO_l2_4loss"
log_dir = os.path.join("./1024_CREAD_RAG/tensorboard_logs", f"run_{time_str}", f"{tips}")
log_file_path = os.path.join(log_dir, "metrics_log.txt")
writer = SummaryWriter(log_dir=log_dir)

if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default='configs/cremad.json')
    
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
    merge_alpha = cfg['train']['merge_alpha']

    lower_a = cfg['train']['lower_a']
    upper_a = cfg['train']['upper_a']
    lower_v = cfg['train']['lower_v']
    upper_v = cfg['train']['upper_v']

    logger, log_file, exp_id = create_logger(cfg, local_rank)

    # ----- SET DATALOADER -----
    train_dataset = CramedDataset(config, mode='train')
    test_dataset = CramedDataset(config, mode='test')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                             num_workers=cfg['test']['num_workers'], pin_memory=True)
    
    
    
    global s_a
    global s_v
    s_a = 1.0
    s_v = 1.0
    count_add_v = 0
    count_add_a = 0
    # ----- MODEL -----
    # 通常在 AudioVideo 模块中定义，可能用于处理音频和视频数据的分类任务
    # 音频信号可能包含说话者的语气，而视频信号可能提供面部表情信息
    # model = AVShareClassifier(config=cfg)

    
    gs = GSPlugin()
    lr_adjust = config['train']['optimizer']['lr']
    epoch_thre = config['train']['epoch_thre']
    thre_sample = config['train']['thre_sample']
    thre_sample_min = config['train']['thre_sample_min']
    warmup_epochs = config['train']['warmup_epochs']
    tmp = config['train']['temperature1']
    logits_ratio = config['train']['logits_ratio']
    merge_alpha_new_v = config['train']['merge_alpha_new_v']
    merge_alpha_new_a = config['train']['merge_alpha_new_a']
    min_alpha = config['train']['min_alpha']

    random_pick_a = config['train']['random_pick_a']
    random_pick_v = config['train']['random_pick_v']
    top_a = config['train']['top_a']
    top_v = config['train']['top_v']
    down_a = config['train']['down_a']
    down_v = config['train']['down_v']

    new_encoder = config['train']['new_encoder']
    all_updata_epoch = config['train']['all_updata_epoch']

    dynamic_epoch_list = config['train']['dynamic_epoch']
    dynamic_epoch_tmp_list = config['train']['dynamic_epoch_tmp']
    decay = config['train']['decay']


    # optimizer = optim.SGD(model.parameters(), lr=lr_adjust,
    #                       momentum=config['train']['optimizer']['momentum'],
    #                       weight_decay=config['train']['optimizer']['wc'])
    ## map 函数将 id 函数应用于 rein_network1 的每一个参数。
    ## id 函数返回对象的唯一标识符（内存地址），这是一个整数，确保每个对象的唯一性。
    ### TODO:1017 ORI is below

    for dynamic_epoch in dynamic_epoch_list:
        for dynamic_epoch_tmp in dynamic_epoch_tmp_list:
            model = AVClassifier(config=cfg)
            model = model.cuda()
            model.apply(weight_init)
            optimizer = optim.SGD(model.parameters(), lr=lr_adjust, momentum=config['train']['optimizer']['momentum'], weight_decay=config['train']['optimizer']['wc'])
            # 每当达到patience步数后，将学习率降低到原来的10 %。
            scheduler = optim.lr_scheduler.StepLR(optimizer, config['train']['lr_scheduler']['patience'], 0.1)
            best_acc = 0
            
            db_file_paths = config['db_file_paths']
            retri_db = read_db(db_file_paths)
            
            for epoch in range(cfg['train']['epoch_dict']):
                logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

                scheduler.step()

                
                model, ratio_a, t_a, t_v, sample_more_a, data_more_o_v_sum = train_audio_video(epoch, train_loader, model, optimizer, logger, 
                                                                                                count_add_a, count_add_v, gs, merge_alpha, 
                                                                                                retri_db = retri_db, epoch_thre = epoch_thre, 
                                                                                                thre_sample= thre_sample, thre_sample_min=thre_sample_min, 
                                                                                                warmup_epochs=warmup_epochs, tmp = tmp, lower_a= lower_a , 
                                                                                                upper_a = upper_a, lower_v=lower_v , upper_v = upper_v, 
                                                                                                logits_ratio = logits_ratio, merge_alpha_new_a = merge_alpha_new_a, 
                                                                                                merge_alpha_new_v = merge_alpha_new_v, min_alpha = min_alpha,
                                                                                                random_pick_a=random_pick_a , top_a=top_a , down_a=down_a,
                                                                                                random_pick_v=random_pick_v , top_v=top_v , down_v=down_v, new_encoder = new_encoder, all_updata_epoch = all_updata_epoch)
                
                if epoch >= dynamic_epoch and (epoch % dynamic_epoch_tmp) == 0:
                    retri_db = make_new_retri_db(model, train_loader, retri_db, epoch, decay)

                acc, v_a, v_v = val(epoch, test_loader, model, logger, merge_alpha, writer)
                
                logger.info(('ratio: {ratio_a:.3f}, sample_more_a:{sample_more_a:.3f}, sample_more_v:{sample_more_v:.3f}').format(ratio_a=ratio_a, sample_more_a=sample_more_a, sample_more_v=data_more_o_v_sum))
                
                if acc > best_acc:
                    best_acc = acc
                    print('Find a better model and save it!')
                    logger.info('Find a better model and save it!')
                    m_name = cfg['visual']['name'] + '_' + cfg['text']['name']
                    # torch.save(model.state_dict(), f'/data/lxe/multimodel/NeurIPS24-LFM-main/Train_RA/100_30_Crema_best_model_{thre_sample}_{epoch}_normal.pth')
