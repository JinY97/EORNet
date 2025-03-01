'''
Author: JinYin
Date: 2023-05-25 17:07:14
LastEditors: Yin Jin
LastEditTime: 2023-10-20 21:02:47
'''
import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--EEG_path', type=str, default="/hdd/yj_data/01_data/02_Denoisenet/EEG_shuffle.npy")
    parser.add_argument('--NOS_path', type=str, default="/hdd/yj_data/01_data/02_Denoisenet/EOG_shuffle.npy")
    parser.add_argument('--denoise_network', type=str, default='EORNet')
    parser.add_argument('--save_dir', type=str, default= '/hdd/yj_data/02_result/01_EORNet/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)

    opts = parser.parse_args()
    return opts

def get_name(opts):
    name = '{}_{}_{}'.format(opts.denoise_network, opts.epochs, opts.batch_size)
    return name
