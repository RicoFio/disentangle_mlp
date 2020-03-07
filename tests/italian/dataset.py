import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import torch

normalize_fiw   = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def get_data_loader(img_size, train_folder, test_folder, batch_size_train, batch_size_test, num_workers_train, num_workers_test):
    my_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize_fiw
    ])
    train_dataset = datasets.ImageFolder(root=train_folder, transform=my_transform)
    test_dataset = datasets.ImageFolder(root=test_folder, transform=my_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers_train)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers_test)

    return train_loader, test_loader