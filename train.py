import os
import time
import json
import argparse
import logging
import calendar
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from GeoDTR import GeoDTR
from datasets.cvusa import USADataset
from datasets.cvact import ACTDataset
from utils import softMarginTripletLoss,\
     CFLoss, save_model, WarmupCosineSchedule, validatenp


def GetBestModel(path):
    all_files = os.listdir(path)
    config_files =  list(filter(lambda x: x.startswith('epoch_'), all_files))
    config_files = sorted(list(map(lambda x: int(x.split("_")[1]), config_files)), reverse=True)
    best_epoch = config_files[0]
    return os.path.join('epoch_'+str(best_epoch), 'trans_'+str(best_epoch)+'.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--save_suffix", type=str, default='test', help='name of the model at the end')
    parser.add_argument("--data_dir", type=str, default='../scratch', help='dir to the dataset')
    parser.add_argument('--dataset', default='CVUSA', choices=['CVUSA', 'CVACT'], help='which dataset to use') 
    parser.add_argument("--n_des", type=int, default=8, help='number of descriptors')
    parser.add_argument("--TR_heads", type=int, default=8, help='number of heads in Transformer')
    parser.add_argument("--TR_layers", type=int, default=8, help='number of layers in Transformer')
    parser.add_argument("--TR_dim", type=int, default=2048, help='dim of FFD in Transformer')
    parser.add_argument("--dropout", type=float, default=0.3, help='dropout in Transformer')
    parser.add_argument("--gamma", type=float, default=10.0, help='value for gamma')
    parser.add_argument("--weight_decay", type=float, default=0.03, help='weight decay value for optimizer')
    parser.add_argument('--pt', default=False, action='store_true', help='turn off polar transformation')
    parser.add_argument('--cf', default=False, action='store_true', help='counter factual loss')
    parser.add_argument('--verbose', default=True, action='store_false', help='turn on progress bar')
    parser.add_argument('--layout_sim', default='strong', choices=['strong', 'weak', 'none'], help='layout simulation strength') 
    parser.add_argument('--sem_aug', default='strong', choices=['strong', 'weak', 'none'], help='semantic augmentation strength') 

    opt = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    batch_size = opt.batch_size
    number_of_epoch = opt.epochs
    learning_rate = opt.lr
    gamma = opt.gamma

    hyper_parameter_dict = vars(opt)
    
    logger.info("Configuration:")
    for k, v in hyper_parameter_dict.items():
        print(f"{k} : {v}", flush=True)

    # generate time stamp
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    save_name = f"{ts}_{opt.model}_{opt.dataset}_{opt.cf}_{opt.pt}_{opt.save_suffix}"
    print("save_name : ", save_name, flush=True)
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    else:
        logger.info("Note! Saving path existed !")

    with open(os.path.join(save_name,f"{ts}_parameter.json"), "w") as outfile:
        json.dump(hyper_parameter_dict, outfile, indent=4)

    writer = SummaryWriter(save_name)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.dataset == "CVUSA":
        dataloader = DataLoader(
            USADataset(
                data_dir = opt.data_dir, 
                layout_simulation = opt.layout_sim, 
                sematic_aug = opt.sem_aug, 
                mode = 'train', 
                is_polar = opt.pt),
            batch_size = batch_size, 
            shuffle = True, 
            num_workers = 8)

        validateloader = DataLoader(
            USADataset(
                data_dir = opt.data_dir,
                layout_simulation = 'none', 
                sematic_aug = 'none', 
                mode = 'val', 
                is_polar = opt.pt),
            batch_size = batch_size, 
            shuffle = False, 
            num_workers = 8)
    elif opt.dataset == "CVACT":
        #train
        dataloader = DataLoader(
            ACTDataset(
                data_dir = opt.data_dir, 
                layout_simulation = opt.layout_sim, 
                sematic_aug = opt.sem_aug, 
                is_polar = opt.pt, 
                mode = 'train'), 
            batch_size = batch_size, 
            shuffle = True, 
            num_workers = 8)

        #val
        validateloader = DataLoader(
            ACTDataset(
                data_dir = opt.data_dir,
                layout_simulation = 'none', 
                sematic_aug = 'none', 
                is_polar = opt.pt, 
                mode = 'val'), 
            batch_size = batch_size, 
            shuffle = False, 
            num_workers = 8)

    model = GeoDTR(n_des=opt.n_des, 
                   tr_heads=opt.TR_heads, 
                   tr_layers=opt.TR_layers, 
                   dropout = opt.dropout, 
                   d_hid = opt.TR_dim, 
                   is_polar = opt.pt)
    embedding_dims = opt.n_des * 512
    model = nn.DataParallel(model)
    model.to(device)

    #set optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = opt.weight_decay, eps = 1e-6)
    lrSchedule = WarmupCosineSchedule(optimizer, 5, number_of_epoch)

    start_epoch = 0

    # Start training
    logger.info("start training...")
    best_epoch = {'acc':0, 'epoch':0}
    for epoch in range(start_epoch, number_of_epoch):

        logger.info(f"start epoch {epoch}")
        epoch_triplet_loss = 0
        if opt.cf:
            epoch_cf_loss = 0
        if opt.intra:
            epoch_it_loss = 0

        model.train() # set model to train
        for batch in tqdm(dataloader, disable = opt.verbose):

            optimizer.zero_grad()

            sat = batch['satellite'].to(device)
            grd = batch['ground'].to(device)

            if opt.cf:
                sat_global, grd_global, fake_sat_global, fake_grd_global = model(sat, grd, opt.cf)
            else:
                sat_global, grd_global = model(sat, grd, opt.cf)
            triplet_loss = softMarginTripletLoss(sate_vecs = sat_global, pano_vecs = grd_global, loss_weight = gamma)
            loss = triplet_loss

            epoch_triplet_loss += loss.item()
            
            if opt.cf:# calculate CF loss
                CFLoss_sat= CFLoss(sat_global, fake_sat_global)
                CFLoss_grd = CFLoss(grd_global, fake_grd_global)
                CFLoss_total = (CFLoss_sat + CFLoss_grd) / 2.0
                loss += CFLoss_total
                epoch_cf_loss += CFLoss_total.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # adjust lr
        lrSchedule.step()

        logger.info(f"Summary of epoch {epoch}")
        print(f"===============================================", flush=True)
        print("---------loss---------", flush=True)
        current_triplet_loss = float(epoch_triplet_loss) / float(len(dataloader))
        print(f"Epoch {epoch} TRI_Loss: {current_triplet_loss}", flush=True)
        writer.add_scalar('triplet_loss', current_triplet_loss, epoch)
        if opt.cf:
            current_cf_loss = float(epoch_cf_loss) / float(len(dataloader))
            print(f"Epoch {epoch} CF_Loss: {current_cf_loss}", flush=True)
            writer.add_scalar('cf_loss', current_cf_loss, epoch)
        
        if opt.intra:
            current_intra_loss = float(epoch_it_loss) / float(len(dataloader))
            print(f"Epoch {epoch} intra loss: {current_intra_loss}", flush=True)
            writer.add_scalar('intra_loss', current_intra_loss, epoch)
            
        print("----------------------", flush=True)

        # Testing phase
        sat_global_descriptor = np.zeros([8884, embedding_dims])
        grd_global_descriptor = np.zeros([8884, embedding_dims])
        val_i = 0

        model.eval()
        with torch.no_grad():
            for batch in tqdm(validateloader, disable = opt.verbose):
                sat = batch['satellite'].to(device)
                grd = batch['ground'].to(device)

                sat_global, grd_global = model(sat, grd, is_cf=False)
                sat_global_descriptor[val_i: val_i + sat_global.shape[0], :] = sat_global.detach().cpu().numpy()
                grd_global_descriptor[val_i: val_i + grd_global.shape[0], :] = grd_global.detach().cpu().numpy()

                val_i += sat_global.shape[0]

            valAcc = validatenp(sat_global_descriptor, grd_global_descriptor)
            logger.info("validation result")
            print(f"------------------------------------", flush=True)
            try:
                #print recall value
                top1 = valAcc[0, 1]
                print('top1', ':', valAcc[0, 1] * 100.0, flush=True)
                print('top5', ':', valAcc[0, 5] * 100.0, flush=True)
                print('top10', ':', valAcc[0, 10] * 100.0, flush=True)
                print('top1%', ':', valAcc[0, -1] * 100.0, flush=True)
                # write to tensorboard
                writer.add_scalars('validation recall@k',{
                    'top 1':valAcc[0, 1],
                    'top 5':valAcc[0, 5],
                    'top 10':valAcc[0, 10],
                    'top 1%':valAcc[0, -1]
                }, epoch)
            except:
                print(valAcc, flush=True)

            # save best model
            if top1 > best_epoch['acc']:
                best_epoch['acc'] = top1
                best_epoch['epoch'] = epoch
                save_model(save_name, model, optimizer, lrSchedule, epoch, last=False)

            # save last model
            save_model(save_name, model, optimizer, lrSchedule, epoch, last=True)
            print(f"=================================================", flush=True)

    # get the best model and recall
    print("best acc : ", best_epoch['acc'], flush=True)
    print("best epoch : ", best_epoch['epoch'], flush=True)
