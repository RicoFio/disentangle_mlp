import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dataset import *
from model import *
from tqdm import tqdm

from envsetter import EnvSetter
from fid import get_fid
from logger import Logger

opt = EnvSetter("vae")
logger = Logger(opt.log_path, opt)
save_path = opt.save_path

torch.manual_seed(opt.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_data_loader(opt)

model = VAE(opt=opt)
model = torch.nn.DataParallel(model)
model = model.to(device)
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in tqdm(enumerate(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch.to(device), data, mu.to(device), logvar.to(device))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    # Generate Samples
    avg_loss = train_loss / len(train_loader.dataset)
    return avg_loss

if __name__ == "__main__":
    start_epoch = 0
    if opt.load_path:
        checkpoint = torch.load(opt.load_path)
        model.load_state_dict(checkpoint['VAE_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    if opt.to_train:
        for epoch in tqdm(range(start_epoch, opt.epochs)):
            avg_loss = train(epoch)
            with torch.no_grad():
                # First thing save
                torch.save({
                    'epoch': epoch + 1,
                    'VAE_model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()}, opt.model_path + f"model_{str(epoch+1)}.tar"))

                # Calculate FID
                fn = lambda x: model.module.decode(x).cpu()
                generate_fid_samples(fn, epoch, opt.n_samples, opt.n_hidden, opt.fid_path_recons, device=device)
                fid = get_fid(opt.fid_path_recons, fid_path_pretrained)
                print('====> Epoch: {} Average loss: {:.4f} FID: {:.4f}'.format(
                    epoch, avg_loss))

                # Log results
                logger.log({
                    "Epoch": epoch, 
                    "Avg Loss": avg_loss, 
                    "FID": fid
                    })
    elif opt.fid:
        raise NotImplementedError
        # Generate samples

        # Calculate FID

        # Return FID and free up space
        
    
