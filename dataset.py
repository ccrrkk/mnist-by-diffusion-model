# dataset.py - 使用torchvision加载MNIST数据集
import torch
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size, seed=1, data_dir='./data'):
    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1], shape: (1,28,28)
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # 划分验证集
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader