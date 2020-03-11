import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torchvision.utils as utils

from dataset import *
from model import *

from tqdm import tqdm
from envsetter import EnvSetter

from helper_functions import *

opt = EnvSetter("gan").get_parser()

# Set-up variables
workers = opt.num_workers
batch_size = opt.batch_size_training
image_size = opt.img_size
nc = opt.input_channels
nz = opt.n_z
ngf = opt.n_hidden
ndf = opt.n_hidden
beta1 = opt.beta
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the generator
netG = Generator_celeba(opt).to(device)
netD = Discriminator_celeba(opt).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and opt.use_gpus:
    netG = nn.DataParallel(netG, opt.use_gpus)
    netD = nn.DataParallel(netD, opt.use_gpus)

# Init weights to mean=0, stdev=0.2.
netG.apply(weights_init)
netD.apply(weights_init)

# Generator
optimizerG = optim.Adam(model.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)

# Data loaders
train_loader, val_loader, test_loader = get_data_loader(opt)

def train():
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(batch_size, nz, device=device)

    avg_loss_G = 0 
    avg_loss_D = 0

    for i, data in enumerate(train_loader, 0):

        # create labels 
        fake_label = np.random.choice(a=[0.1,0.9], p=[0.95, 0.05])
        real_label = np.random.choice(a=[0.1,0.9], p=[0.05, 0.95])

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output, _ = netD(real_cpu)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output, _ = netD(fake.detach())
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output, _ = netD(fake)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % opt.log_interval == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(train_loader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        avg_errG += errG
        avg_errD += errD

    avg_errG = avg_errG / len(train_loader.dataset)
    avg_errD = avg_errG / len(train_loader.dataset)

    return avg_errG, avg_errD


if __name__ == "__main__":
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    start_epoch = 0
    if opt.load_path:
        checkpoint = torch.load(opt.load_path)
        start_epoch = checkpoint['epoch']
        netG.load_state_dict(checkpoint['netG'])
        netD.load_state_dict(checkpoint['netD'])
        optimizerG.load_state_dict(checkpoint['G_trainer'])
        optimizerD.load_state_dict(checkpoint['D_trainer'])

    if opt.to_train:
        for epoch in range(start_epoch, opt.epochs):
           avg_errG, avg_errD = train()
            with torch.no_grad():
            # Save Model
            torch.save({
                'epoch': epoch + 1,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                'G_trainer': optimizerG.state_dict(),
                'D_trainer': optimizerD.state_dict()}, save_path.replace('%',str(epoch+1)))

            # Calculate FID
            fn = lambda x: netG(x).detach().cpu()
            generate_fid_samples(fn, epoch, opt.n_samples, opt.n_hidden, opt.fid_path_recons, device=device)
            fid = get_fid(opt.fid_path_recons, fid_path_pretrained)

            # Output stats
            print('====> Epoch: {} Average loss: {:.4f} FID: {:.4f}'.format(
                epoch, avg_loss))

            # Log results
            logger.log({
                "Epoch": epoch, 
                "Avg Loss G": avg_errG, 
                "Avg Loss E": avg_errD,
                "FID": fid
                })
                
    elif:
        raise NotImplementedError











