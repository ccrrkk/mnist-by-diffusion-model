# sample.py - 采样与可视化
import torch
import matplotlib.pyplot as plt
from model import UNet
from diffusion import DenoiseDiffusion

def show_sample(images, texts,save_fig_path='sample.png'):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for text, f, img in zip(texts, figs, images):
        f.imshow(img.view(28, 28), cmap='gray')
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        f.text(0.5, 0, text, ha='center', va='bottom', fontsize=12, color='white', backgroundcolor='black')
    plt.savefig(save_fig_path)
    plt.show()

def sample(device, save_dir,save_fig_path,class_label=7):
    # 采样过程: 从随机噪声开始，通过训练好的UNet逐步去噪生成图像
    # 1. 初始化随机噪声 (形状: [batch_size, channels, height, width])
    #    这里使用标准正态分布生成随机噪声，对应扩散过程的最终状态
    xt = torch.randn((1, 1, 28, 28), device=device)
    y = torch.tensor([class_label], device=device)  # 指定生成数字的类别标签
    # 存储中间结果用于可视化
    images, texts = [], []
    # 2. 创建并加载训练好的UNet模型
    u_net = UNet(
        image_channels=1,        # 输入通道数 (灰度图)
        n_channels=16,          # 基础通道数
        ch_mults=[1, 2, 2],     # 特征图倍数序列
        is_attn=[False, False, False],  # 注意力机制配置
        n_blocks=1,              # 块数量
        num_class=10            # 类别数量
    ).to(device)
    # 加载训练好的模型权重
    u_net.load_state_dict(torch.load(save_dir, map_location=device))
    u_net.eval()  
    # 3. 创建去噪扩散模型实例
    dm = DenoiseDiffusion(u_net, 1000, device=device)
    # 4. 反向扩散过程：从t=999到t=0逐步去噪
    with torch.no_grad():  # 重要：在采样时不需要计算梯度
        for t in reversed(range(1000)):
            # 使用训练好的模型从x_t预测x_{t-1}
            xt_1 = dm.p_sample(xt, torch.tensor([t]).to(device), y)
            xt = xt_1  # 更新当前状态
            # 每100步保存一次中间结果用于可视化
            if (t + 1) % 100 == 1:  
                # 将张量转移到CPU并分离计算图，准备保存
                image_tensor = xt.view(1, 28, 28).to('cpu').detach()
                images.append(image_tensor)
                texts.append(t + 1)  # 记录时间步
    # 5. 处理并展示结果
    # 将所有保存的图像堆叠成一个张量
    images_ = torch.stack(images, dim=0)
    # 显示采样过程的可视化结果
    show_sample(images_, texts,save_fig_path=save_fig_path)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_dir = '/opt/data/private/mnist-by-diffusion-model/ConditionalDiffusion/ckpt/u_net_20251004_120632.pt'
    for class_label in range(10):
        print(f'Sampling for class {class_label}...')
        save_fig_path = f'sample_class_{class_label}.png'
        sample(device, save_dir, save_fig_path, class_label=class_label)

