import torch
import numpy as np
import os
import math
from torch.optim.lr_scheduler import LambdaLR
import json


def ReadConfig(path):
    all_files = os.listdir(path)
    config_file =  list(filter(lambda x: x.endswith('parameter.json'), all_files))
    with open(os.path.join(path, config_file[0]), 'r') as f:
        p = json.load(f)
        return p


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def softMarginTripletLoss(sate_vecs, pano_vecs, loss_weight=10.0, hard_topk_ratio=1.0):
    dists = 2.0 - 2.0 * torch.matmul(sate_vecs, pano_vecs.permute(1, 0))  # Pairwise matches within batch
    pos_dists = torch.diag(dists)
    N = len(pos_dists)
    diag_ids = np.arange(N)
    num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if hard_topk_ratio < 1.0 else N * (N - 1)

    # Match from satellite to street pano
    triplet_dist_s2p = pos_dists.unsqueeze(1) - dists
    loss_s2p = torch.log(1.0 + torch.exp(loss_weight * triplet_dist_s2p))
    loss_s2p[diag_ids, diag_ids] = 0.0  # Ignore diagnal losses

    if hard_topk_ratio < 1.0:  # Hard negative mining
        loss_s2p = loss_s2p.view(-1)
        loss_s2p, s2p_ids = torch.topk(loss_s2p, num_hard_triplets)
    loss_s2p = loss_s2p.sum() / float(num_hard_triplets)

    # Match from street pano to satellite
    triplet_dist_p2s = pos_dists - dists
    loss_p2s = torch.log(1.0 + torch.exp(loss_weight * triplet_dist_p2s))
    loss_p2s[diag_ids, diag_ids] = 0.0  # Ignore diagnal losses

    if hard_topk_ratio < 1.0:  # Hard negative mining
        loss_p2s = loss_p2s.view(-1)
        loss_p2s, p2s_ids = torch.topk(loss_p2s, num_hard_triplets)
    loss_p2s = loss_p2s.sum() / float(num_hard_triplets)
    # Total loss
    loss = (loss_s2p + loss_p2s) / 2.0
    return loss

def CFLoss(vecs, hat_vecs, loss_weight=5.0):
    dists = 2.0 * torch.matmul(vecs, hat_vecs.permute(1, 0)) - 2.0
    cf_dists = torch.diag(dists)
    loss = torch.log(1.0 + torch.exp(loss_weight * cf_dists))
    loss = loss.sum() / vecs.shape[0]
    return loss


def save_model(savePath, model, optimizer, scheduler, epoch, last=True):
    if last == True:
        save_folder_name = "epoch_last"
        model_name = "epoch_last.pth"
    else:
        save_folder_name = f"epoch_{epoch}"
        model_name = f'epoch_{epoch}.pth'
    modelFolder = os.path.join(savePath, save_folder_name)
    if os.path.isdir(modelFolder):
        pass
    else:
        os.makedirs(modelFolder)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(modelFolder, model_name))


def validatenp(sat_global_descriptor, grd_global_descriptor):
    dist_array = 2.0 - 2.0 * np.matmul(sat_global_descriptor, grd_global_descriptor.T)
    top1_percent = int(dist_array.shape[0] * 0.01) + 1
    val_accuracy = np.zeros((1, top1_percent))
    for i in range(top1_percent):
        accuracy = 0.0
        data_amount = 0.0
        for k in range(dist_array.shape[0]):
            gt_dist = dist_array[k,k]
            prediction = np.sum(dist_array[:, k] < gt_dist)
            if prediction < i:
                accuracy += 1.0
            data_amount += 1.0
        accuracy /= data_amount
        val_accuracy[0, i] = accuracy
    return val_accuracy