############################
# Imports
############################
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import argparse
import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import DataParallel
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dataset import *
from model import *
from tqdm import tqdm

from fid import get_fid
from logger import Logger
from envsetter import EnvSetter
from helper_functions import *

############################

# Globals

# function to add to JSON 
opt = EnvSetter("vaegan").get_parser()
logger = Logger(opt.log_path, opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(opt.seed)

# Load data 
train_loader, val_loader, test_loader = get_data_loader(opt)

netEG = VAE(opt=opt)
netEG = DataParallel(netEG.to(device))
netD = Discriminator_celeba(opt)
netD = DataParallel(netD.to(device))

netEG.apply(weights_init)
netD.apply(weights_init)

optimizerEG = optim.Adam(netEG.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)

# Initialize BCELoss function
criterion = nn.BCELoss()
#############################
# Reconstruction + KL divergence losses summed over all elements and batch
# def reconstruction_loss(recon_x, x, mu, logvar):

#     MSE = F.mse_loss(recon_x, x, reduction='sum')

#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return MSE + opt.beta * KLD

def KLD(mu, logvar):
    return opt.beta * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

def SIM(sim_recon, sim_real):
    SIM = F.mse_loss(sim_recon, sim_real, reduction='sum')
    return 0.1 * SIM

def reconstruction_loss(recon_x, x):

    MSE = F.mse_loss(recon_x, x, reduction='sum')

    return MSE 

def train(epoch):
    netD.train()
    netEG.train()
    
    recon_enc_loss = 0
    train_recon_enc_loss = 0
    train_recon_dec_loss = 0
    avg_dis_loss = 0

    for batch_idx, (data, _) in tqdm(enumerate(train_loader)):
        # create labels 
        fake_label = np.random.choice(a=[0.1,0.9], p=[0.95, 0.05])
        real_label = np.random.choice(a=[0.1,0.9], p=[0.05, 0.95])
        data = data.to(device)

        ### Discriminator ###

        netD.zero_grad()

        label = torch.full((data.size()[0],), real_label, device=device)
        # Forward pass real batch through D
        output, sim_real = netD(data)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()

        avg_dis_loss += output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(data.size()[0], 128, device=device)
        # Generate fake image batch with G
        fake = netEG.module.decode(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output, _ = netD(fake.detach())
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)

        # Calculate the gradients for this batch
        errD_fake.backward()
        # Update D
        optimizerD.step()

        ### Decoder ###

        netEG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost

        # encoder to reuires grad = False
        netEG.module.features.requires_grad = False
        netEG.module.x_to_mu.requires_grad = False
        netEG.module.x_to_logvar.requires_grad = False
        netEG.module.preprocess.requires_grad = True
        netEG.module.deconv1.requires_grad = True
        netEG.module.act1.requires_grad = True
        netEG.module.deconv2.requires_grad = True
        netEG.module.act2.requires_grad = True
        netEG.module.deconv3.requires_grad = True
        netEG.module.act3.requires_grad = True
        netEG.module.deconv4.requires_grad = True
        netEG.module.activation.requires_grad = True       
        recon_batch, mu, logvar = netEG(data)

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output_fake, _ = netD(fake)

        # should add this too
        output_recon, sim_recon = netD(recon_batch)
       
        # Calculate G's loss based on this output
        errG_fake = criterion(output_fake, label)
        # errG_recon = criterion(output_recon, label)
        
        # Calculate gradients for G
        errG_fake.backward()
        #errG_recon.backward()

        sim_loss = SIM(sim_real=sim_real.to(device), sim_recon=sim_recon.to(device))
        sim_loss.backward()
        loss = reconstruction_loss(recon_x=recon_batch.to(device), x=data, is_gen=True)
        loss.backward()
        optimizerEG.step()

        ### Encoder ###
        netEG.zero_grad()

        netEG.module.features.requires_grad = True
        netEG.module.x_to_mu.requires_grad = True
        netEG.module.x_to_logvar.requires_grad = True
        netEG.module.preprocess.requires_grad = False
        netEG.module.deconv1.requires_grad = False
        netEG.module.act1.requires_grad = False
        netEG.module.deconv2.requires_grad = False
        netEG.module.act2.requires_grad = False
        netEG.module.deconv3.requires_grad = False
        netEG.module.act3.requires_grad = False
        netEG.module.deconv4.requires_grad = False
        netEG.module.activation.requires_grad = False

        recon_batch, mu, logvar = netEG(data)

        kld = KLD(mu.to(device), logvar.to(device))
        kld.backward()
        loss = reconstruction_loss(recon_x=recon_batch.to(device), x=data)
        loss.backward()

        train_recon_enc_loss += loss.item()
        train_recon_dec_loss += loss.item()

        optimizerEG.step()
        break

    
    avg_recon_enc_loss = train_recon_enc_loss / len(train_loader.dataset)
    avg_recon_dec_loss = train_recon_dec_loss / len(train_loader.dataset)
    avg_dis_loss = avg_dis_loss / len(train_loader.dataset)

    return avg_recon_enc_loss, avg_recon_dec_loss, avg_dis_loss

if __name__ == "__main__":
#    set_up_globals()

    start_epoch = 0
    if opt.load_path:
        checkpoint = torch.load(opt.load_path)
        start_epoch = checkpoint['epoch']
        netEG.module.load_state_dict(checkpoint['encoder_decoder_model'])
        netD.load_state_dict(checkpoint['discriminator_model'])
        optimizerEG.load_state_dict(checkpoint['encoder_decoder_optimizer'])
        optimizerD.load_state_dict(checkpoint['discriminator_optimizer'])

    if opt.to_train:
        for epoch in tqdm(range(start_epoch, opt.epochs)):
            enc_loss, dec_loss, dis_loss = train(epoch)
            with torch.no_grad():
                torch.save({
                    'epoch': epoch + 1,
                    "encoder_decoder_model": netEG.module.state_dict(),
                    "discriminator_model": netD.state_dict(),
                    'encoder_decoder_optimizer': optimizerEG.state_dict(),
                    'discriminator_optimizer': optimizerD.state_dict(),
                    }, opt.model_path + f"/model_{str(epoch+1)}.tar")
                
                # Calculate FID score
                fn = lambda x: model.module.decode(x).cpu()
                generate_fid_samples(fn, epoch, opt.n_samples, opt.n_hidden, opt.fid_path_recon, device=device)
                fid = get_fid(opt.fid_path_recon, opt.fid_path_pretrained)
                print('====> Epoch: {} Avg Encoder Loss: {:.4f} Avg Decoder Loss: {:.4f} Avg Discriminator Loss: {:.4f} FID: {:.4f}'.format(
                    epoch, enc_loss, dec_loss, dis_loss, fid))

                # Log epoch statistics
                logger.log({
                    "Epoch":epoch, 
                    "Avg Eec Loss": enc_loss, 
                    "Avg Dnc Loss": dec_loss, 
                    "Avg Dis Loss": dis_loss,
                    "FID":fid})
    elif opt.fid:
        raise NotImplementedError
