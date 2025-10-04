# main.py - 项目入口
import torch
from train import train
from sample import sample
import time
import os

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 生成可读时间戳
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = f'ckpt/u_net_{timestamp}.pt'
    os.makedirs('ckpt', exist_ok=True)
    batch_size = 512
    train(device, save_dir, batch_size)
    sample(device, save_dir)

if __name__ == '__main__':
    main()