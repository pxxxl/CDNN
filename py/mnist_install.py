import torch
import torchvision
import torchvision.transforms as transforms
import pickle

# 定义数据转换
transform = transforms.Compose([transforms.ToTensor()])

# 下载并加载训练数据集
train_dataset = torchvision.datasets.MNIST(root='../cache/data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

# 下载并加载测试数据集
test_dataset = torchvision.datasets.MNIST(root='../cache/data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

for data in train_loader:
    train_images, train_labels = data


for data in test_loader:
    test_images, test_labels = data

# to be continued...