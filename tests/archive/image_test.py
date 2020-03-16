import numpy as np
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import torch

from helper_functions import gen_reconstructions

def get_celeba(batch_size, dataset_directory, dataloader_workers):
    train_transformation = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = torchvision.datasets.ImageFolder(dataset_directory, train_transformation)

    # Use sampler for randomization
    training_sampler = torch.utils.data.SubsetRandomSampler(range(len(train_dataset)))

    # Prepare Data Loaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=training_sampler,
                                               pin_memory=True, num_workers=dataloader_workers)

    return train_loader

def currrent_get_celeba(batch_size, dataset_directory, dataloader_workers):
    my_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.ImageFolder(root=dataset_directory, transform=my_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)

    return train_loader


if __name__=="__main__":
    print("Load dataset")
    data_path = r"C:\Users\ricca\Desktop\mlp\tests\current\data\celebAtest"
    train_loader_1 = get_celeba(24, data_path, 1)
    train_loader_2 = currrent_get_celeba(24, data_path, 1)
    fn = lambda x: x
    gen_reconstructions(fn, train_loader_1, 1, r"C:\Users\ricca\Desktop\mlp\tests\current\data\celebAtestResults")
    gen_reconstructions(fn, train_loader_2, 2, r"C:\Users\ricca\Desktop\mlp\tests\current\data\celebAtestResults")