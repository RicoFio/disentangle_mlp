import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import torch

normalize_birds = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
normalize_mnist = transforms.Normalize(mean=[0.1307], std=[0.3081])
normalize_fiw   = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def get_data_loader(opt):
    if opt.dataset == "birds":
        my_transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            normalize_birds
        ])
        train_dataset = datasets.ImageFolder(root=opt.image_root, transform=my_transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        test_loader = None
        val_loader = None

    elif opt.dataset == "mnist":
        my_transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            normalize_mnist
        ])
        train_loader = DataLoader(datasets.MNIST(opt.image_root, train=True, download=True, transform=my_transform),
                                    batch_size=opt.batch_size, shuffle=True)
        test_loader = None
        val_loader = None

    elif opt.dataset == "celebA" or opt.dataset == "celebA_reduced":
        my_transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
            normalize_fiw
        ])
        train_dataset = datasets.ImageFolder(root=opt.image_root_train, transform=my_transform)
        val_dataset = datasets.ImageFolder(root=opt.image_root_val, transform=my_transform)
        test_dataset = datasets.ImageFolder(root=opt.image_root_test, transform=my_transform)

        train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size_train, shuffle=True, num_workers=opt.num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size_val, shuffle=False, num_workers=opt.num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size_test, shuffle=False, num_workers=opt.num_workers)

    return train_loader, val_loader, test_loader