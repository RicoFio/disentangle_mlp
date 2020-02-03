""" 
dataset class 

torch.utils.data.Dataset is an abstract class representing a dataset

we inherit dataset and override __len__ and __getitem__

"""

"""
This is from the PyTorch dataloader tutorial at pytorch.org! Thanks!
"""
from __future__ import print_function, division
import os
import torch 
import pandas as pd
from skimage import io, transform 
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils 

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion() # interactive

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        args:

        """
        
    def __len__(self):
        return len(self. ) 
    
    def __getitem__(self, idx):
