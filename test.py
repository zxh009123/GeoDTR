import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from GeoDTR import GeoDTR
from datasets.cvusa import USADataset
from datasets.cvact import ACTDataset
from utils import ReadConfig, validatenp


args_do_not_overide = ['data_dir', 'verbose', 'dataset']

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def GetBestModel(path):
    all_files = os.listdir(path)
    if "epoch_last" in all_files:
        all_files.remove("epoch_last")
    config_files =  list(filter(lambda x: x.startswith('epoch_'), all_files))
    config_files = sorted(list(map(lambda x: int(x.split("_")[1]), config_files)), reverse=True)
    best_epoch = config_files[0]
    return os.path.join('epoch_'+str(best_epoch), 'epoch_'+str(best_epoch)+'.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='../scratch/', help='dir to the dataset')
    parser.add_argument('--dataset', default='CVUSA', choices=['CVUSA', 'CVACT'], help='choose between CVUSA or CVACT')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--model_path', type=str, help='path to model weights')

    opt = parser.parse_args()

    config = ReadConfig(opt.model_path)
    for k,v in config.items():
        if k in args_do_not_overide:
            continue
        setattr(opt, k, v)
    print(opt, flush=True)

    batch_size = opt.batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if opt.dataset == 'CVACT':
        data_path = os.path.join(opt.data_dir, 'CVACT')
        dataset = ACTDataset(data_dir = data_path, layout_simulation='none', sematic_aug='none', is_polar=opt.pt, mode='val')
        validateloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    if opt.dataset == 'CVUSA':
        data_path = os.path.join(opt.data_dir, 'CVUSA', 'dataset')
        dataset = USADataset(data_dir = data_path, layout_simulation='none', sematic_aug='none', mode='val', is_polar=opt.pt)
        validateloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    print("number of test samples : ", len(dataset), flush=True)

    model = GeoDTR(n_des = opt.n_des, 
                   tr_heads = opt.TR_heads, 
                   tr_layers = opt.TR_layers, 
                   dropout = opt.dropout, 
                   d_hid = opt.TR_dim, 
                   is_polar = opt.pt)
    embedding_dims = opt.n_des * 512
    model = nn.DataParallel(model)
    model.to(device)

    best_model = GetBestModel(opt.model_path)
    best_model = os.path.join(opt.model_path, best_model)
    print("loading model : ", best_model, flush=True)
    model.load_state_dict(torch.load(best_model)['model_state_dict'])

    num_params = count_parameters(model)
    print(f"model parameters : {num_params}M", flush=True)

    print("start testing...", flush=True)
    sat_global_descriptor = np.zeros([len(dataset), embedding_dims])
    grd_global_descriptor = np.zeros([len(dataset), embedding_dims])
    val_i = 0

    model.eval()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings=np.zeros((len(validateloader),1))
    index = 0
    with torch.no_grad():
        for batch in tqdm(validateloader, disable=opt.verbose):
            sat = batch['satellite'].to(device)
            grd = batch['ground'].to(device)

            starter.record()
            sat_global, grd_global = model(sat, grd, is_cf=False)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[index] = curr_time
            index += 1

            sat_global_descriptor[val_i: val_i + sat_global.shape[0], :] = sat_global.detach().cpu().numpy()
            grd_global_descriptor[val_i: val_i + grd_global.shape[0], :] = grd_global.detach().cpu().numpy()

            val_i += sat_global.shape[0]

    mean_syn = np.sum(timings) / len(validateloader)
    print("average inference time : ", mean_syn, flush=True)

    valAcc = validatenp(sat_global_descriptor, grd_global_descriptor)
    print(f"-----------validation result---------------", flush=True)
    try:
        top1 = valAcc[0, 1]
        print('top1', ':', valAcc[0, 1] * 100.0, flush=True)
        print('top5', ':', valAcc[0, 5] * 100.0, flush=True)
        print('top10', ':', valAcc[0, 10] * 100.0, flush=True)
        print('top1%', ':', valAcc[0, -1] * 100.0, flush=True)
    except:
        print(valAcc, flush=True)
    print(f"=================================================", flush=True)
