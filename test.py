"""
-*- coding: utf-8 -*-
@Author : Hollow Handsome
@Time : 2024/11/29 17:26
@File : test.py
"""
import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


device = torch.device("cuda")


image_path = "dataset/flowers-102/jpg/image_01581.jpg"
image = Image.open(image_path)
image = image.convert('RGB')
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.fc = nn.Linear(512, 102)
resnet18.load_state_dict(torch.load("hollow_13.pth"))
model = resnet18.to(device)
print(model)

image = torch.reshape(image,(1,3,224,224))
image = image.to(device)
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))