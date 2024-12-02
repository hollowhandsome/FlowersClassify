"""
-*- coding: utf-8 -*-
@Author : Floo
@Time : 2024/11/29 13:10
@File : FlowersClassify.py
"""

import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


pre_resize = transforms.Resize((224, 224))
pre_totensor = transforms.ToTensor()
pre_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
pre_compose = transforms.Compose([pre_resize, pre_totensor, pre_normalize])


train_data = torchvision.datasets.Flowers102("dataset", split='test', download=True, transform=pre_compose)
test_data = torchvision.datasets.Flowers102("dataset", split='val', download=True, transform=pre_compose)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为{}".format(train_data_size))
print("测试数据集的长度为{}".format(test_data_size))

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)


resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.fc = nn.Linear(512, 102)

hollow = resnet18.to(device)


loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)


learning_rate = 0.001
optimizer = torch.optim.Adam(hollow.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 25

writer = SummaryWriter("logs_train")

start_time = time.time()
for i in range(epoch):
    print("-------------第{}轮开始--------------".format(i+1))
    hollow.train()
    total_train_loss = 0
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = hollow(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        total_train_loss += loss.item()


        end_time = time.time()
        print(end_time - start_time)
        print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    hollow.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = hollow(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss / len(test_dataloader)))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss / len(test_dataloader), total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(hollow.state_dict(), "hollow_{}.pth".format(i))
    print("模型参数{}已保存".format(i))

writer.close()