'''
Author: JinYin
Date: 2023-05-25 17:06:43
LastEditors: Yin Jin
LastEditTime: 2024-03-09 16:31:32
FilePath: /20_CTNet/train_denoisenet.py
Description: 
'''
import os
import time
from tqdm import trange
import torch.nn.functional as F
import torch, numpy as np
import torch.optim as optim

from opts import get_opts
from preprocess.DenoisenetPreprocess import *
from audtorch.metrics.functional import pearsonr
from models.EORNet import FeatureExtractor

def denoise_loss_mse(denoise, clean):      
    loss = torch.nn.MSELoss()
    return loss(denoise, clean)

def cal_SNR(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    PS = np.sum(np.square(truth), axis=-1)  # power of signal
    PN = np.sum(np.square((predict - truth)), axis=-1)  # power of noise
    ratio = PS / PN
    return torch.from_numpy(10 * np.log10(ratio))

def train(opts, model, fold):
    EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data = load_data(opts.EEG_path, opts.NOS_path, fold)
    train_data = EEGwithNoise(EEG_train_data, NOS_train_data, opts.batch_size)
    val_data = EEGwithNoise(EEG_val_data, NOS_val_data, opts.batch_size)
    test_data = EEGwithNoise(EEG_test_data, NOS_test_data, opts.batch_size)
    
    params = []
    params.append({'params': model.parameters()})

    if opts.denoise_network == 'EORNet':        
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99), eps=1e-8)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50], 0.1)
    
    best_val_mse = 1.0
    f = open(opts.save_path + "result.txt", "a+")
    for epoch in range(opts.epochs):
        model.train()
        losses = []
        for batch_id in trange(train_data.len()):
            x, y = train_data.get_batch(batch_id)
            x, y = torch.Tensor(x).to(opts.device).unsqueeze(dim=1), torch.Tensor(y).to(opts.device)
            p = model(x).view(x.shape[0], -1)
            loss = denoise_loss_mse(p, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        
        train_data.shuffle()
        train_loss = torch.stack(losses).mean().item()
        scheduler.step()
        
        model.eval()
        losses = []
        for batch_id in range(val_data.len()):
            x, y = val_data.get_batch(batch_id)
            x, y = torch.Tensor(x).to(opts.device).unsqueeze(dim=1), torch.Tensor(y).to(opts.device)

            with torch.no_grad():
                p = model(x).view(x.shape[0], -1)
                loss = ((p - y) ** 2).mean(dim=-1).sqrt().detach()
                losses.append(loss)
        val_mse = torch.cat(losses, dim=0).mean().item()
        
        model.eval()
        losses = []
        single_acc, single_snr = [], []
        clean_data, output_data, input_data = [], [], []
        for batch_id in range(test_data.len()):
            x, y = test_data.get_batch(batch_id)
            x, y = torch.Tensor(x).to(opts.device).unsqueeze(dim=1), torch.Tensor(y).to(opts.device)
  
            with torch.no_grad():
                p = model(x).view(x.shape[0], -1)
                loss = (((p - y) ** 2).mean(dim=-1).sqrt() / (y ** 2).mean(dim=-1).sqrt()).detach()
                losses.append(loss)
                single_acc.append(pearsonr(p, y))
                single_snr.append(cal_SNR(p, y))

            output_data.append(p.cpu().numpy()), clean_data.append(y.cpu().numpy()), input_data.append(x.cpu().numpy())
        test_rrmse = torch.cat(losses, dim=0).mean().item()
        sum_acc = torch.cat(single_acc, dim=0).mean().item()
        sum_snr = torch.cat(single_snr, dim=0).mean().item()
        
        # 保存最好的结果
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_acc = sum_acc
            best_snr = sum_snr
            best_rrmse = test_rrmse
            print("Save best result")
            f.write("Save best result \n")
            np.save(f"{opts.save_path}/best_input_data.npy", np.concatenate(input_data, axis=0))
            np.save(f"{opts.save_path}/best_output_data.npy", np.concatenate(output_data, axis=0))
            np.save(f"{opts.save_path}/best_clean_data.npy", np.concatenate(clean_data, axis=0))
            torch.save(model, f"{opts.save_path}/best.pth")
        
        print(f"train_loss:{train_loss}")
        print('epoch: {:3d}, val_mse: {:.4f}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(epoch, val_mse, test_rrmse, sum_acc, sum_snr))
        f.write('epoch: {:3d}, val_mse: {:.4f}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}\n'.format(epoch, val_mse, test_rrmse, sum_acc, sum_snr))

    with open(os.path.join('./json_file/01_Denoisenet/EOG_{}_{}_{}.log'.format(opts.denoise_network, opts.depth, opts.e_size)), 'a+') as fp:
        fp.write('fold:{}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(fold, best_rrmse, best_acc, best_snr) + "\n")

def pick_models(opts, data_num=512):
    if opts.denoise_network == "EORNet":
        model = FeatureExtractor(emb_size=64, data_num=512, chan=1, depth=opts.depth, e_size=opts.e_size).to(opts.device)        
         
    else:
        print("model name is error!")
        pass
    
    return model
  
if __name__ == '__main__':
    opts = get_opts()
    opts.depth = 4
    opts.epochs = 50
    opts.batch_size = 32
    opts.e_size = 2
    opts.denoise_network = "EORNet"
    opts.EEG_path = "/hdd/yj_data/01_data/02_Denoisenet/EEG_shuffle.npy"
    opts.NOS_path = "/hdd/yj_data/01_data/02_Denoisenet/EOG_shuffle.npy"
    opts.save_dir = "/hdd/yj_data/02_result/01_EORNet/"
    
    for fold in range(10):
        opts.save_path = "{}/{}_{}/{}_{}_{}/".format(opts.save_dir, opts.denoise_network, opts.depth, opts.denoise_network, opts.e_size, fold)
        if not os.path.exists(opts.save_path):
            os.makedirs(opts.save_path)
        model = pick_models(opts, data_num=512)
        train(opts, model, fold)