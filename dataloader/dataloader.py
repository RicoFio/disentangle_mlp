import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms

class DataLoader:
    def __init__(self, root_dir, batch_size=128, shuffle=True, image_size=64, num_workers=2):
        """
        args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.num_workers = num_workers

    def get_dataloader(self):
        dataset = torchvision.datasets.ImageFolder(root=self.root_dir,
                                                   transform=transforms.Compose([
                                                       transforms.Resize(self.image_size),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                   ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                 shuffle=self.shuffle, num_workers=self.num_workers)
        return dataloader
