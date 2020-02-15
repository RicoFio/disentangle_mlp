import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

normalize_birds = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
normalize_mnist = transforms.Normalize(mean=[0.1307], std=[0.3081])

def get_data_loader(opt):
    if opt.dataset == "birds":
        my_transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            normalize_birds
        ])
        train_dataset = datasets.ImageFolder(root = 'data/images', transform=my_transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    elif opt.dataset == "mnist":
        train_loader = DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
            transform=my_transform),
            batch_size=opt.batch_size, shuffle=True, **opt)

    elif opt.dataset == "celebA":
        raise NotImplementedError

    return train_loader