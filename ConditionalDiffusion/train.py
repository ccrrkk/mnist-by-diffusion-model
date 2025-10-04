# train.py - 训练主循环（适配torchvision MNIST数据集）
import torch
from model import UNet
from diffusion import DenoiseDiffusion
from dataset import get_mnist_loaders

def train(device, save_dir, batch_size=512, seed=1):
    # 1. 数据准备
    # 获取MNIST数据加载器，返回训练集、验证集和测试集
    train_loader, valid_loader, _ = get_mnist_loaders(batch_size, seed)
    # 2. 模型和优化器初始化
    lr = 0.001
    # 创建UNet模型，用于噪声预测
    u_net = UNet(
        image_channels=1,        # 输入通道数 (MNIST是灰度图)
        n_channels=16,          # 基础通道数
        ch_mults=[1, 2, 2],     # 特征图倍数序列
        is_attn=[False, False, False],  # 注意力机制配置
        n_blocks=1,              # 残差块数量
        num_class=10            # 类别数量
    ).to(device)
    # 创建去噪扩散模型，包含前向过程和损失计算
    dm = DenoiseDiffusion(u_net, 1000, device=device)
    opt_dm = torch.optim.Adam(u_net.parameters(), lr=lr)
    # 3. 训练参数设置
    best_score = 1e10        # 初始化最佳验证损失为一个很大的数
    epochs = 100             # 最大训练轮数
    early_stop_time = 0      # 早停计数器
    early_stop_threshold = 40  # 早停阈值
    # 4. 训练循环
    for epoch in range(epochs):
        # 训练阶段
        u_net.train()  
        loss_record = []
        for step, (pic, labels) in enumerate(train_loader):
            pic = pic.to(device)  # 将图片数据移动到设备上，shape: [B, 1, 28, 28]
            labels = labels.to(device)
            # 梯度清零
            opt_dm.zero_grad()
            # 计算扩散模型损失
            # 在扩散模型中，损失通常是预测噪声与真实噪声的MSE
            loss = dm.loss(pic,labels)
            loss_record.append(loss.item())
            # 反向传播
            loss.backward()
            # 参数更新
            opt_dm.step()
        # 打印训练损失
        train_mean_loss = torch.tensor(loss_record).mean()
        print(f'training epoch: {epoch}, mean loss: {train_mean_loss}')
        # 验证阶段
        u_net.eval()  
        loss_record = []
        with torch.no_grad():  # 验证时不需要计算梯度
            for step, (pic, labels) in enumerate(valid_loader):
                pic = pic.to(device)
                labels = labels.to(device) 
                loss = dm.loss(pic,labels)
                loss_record.append(loss.item())
        # 计算验证集平均损失
        mean_loss = torch.tensor(loss_record).mean()
        # 5. 模型保存和早停机制
        if mean_loss < best_score:
            early_stop_time = 0           # 重置早停计数器
            best_score = mean_loss        # 更新最佳分数
            torch.save(u_net.state_dict(), save_dir)  # 保存最佳模型
            print(f'New best model saved with loss: {best_score:.4f}')
        else:
            early_stop_time += 1          # 增加早停计数器
        # 检查是否达到早停条件
        if early_stop_time > early_stop_threshold:
            print(f'Early stopping triggered at epoch {epoch}')
            break
        # 打印早停信息
        print(f'early_stop_time/early_stop_threshold: {early_stop_time}/{early_stop_threshold}, valid loss: {mean_loss:.4f}')