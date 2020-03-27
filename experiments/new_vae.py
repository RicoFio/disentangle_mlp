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
from helper_functions import *
from fid import get_fid
from logger import Logger

opt = EnvSetter("vae").get_parser()
logger = Logger(opt.log_path, opt)
save_path = opt.save_path

torch.manual_seed(opt.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_data_loader(opt)

model = VAE(opt=opt)
model = model.to(device)
model = torch.nn.DataParallel(model)
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

def load_model(path):
    checkpoint = torch.load(path)
    model.module.load_state_dict(checkpoint['VAE_model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

if __name__ == "__main__":
    start_epoch = 0
    if opt.load_path and len(opt.load_path) < 2:
        start_epoch = load_model(opt.load_path[0])

    if opt.to_train:
        for epoch in tqdm(range(start_epoch, opt.epochs)):
            avg_loss = train(epoch)
            with torch.no_grad():
                # First thing save
                torch.save({
                    'epoch': epoch + 1,
                    'VAE_model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()}, opt.model_path + f"model_{str(epoch+1)}.tar")

                # Calculate FID
                fid = "N/A"
                if opt.calc_fid:
                    fn = lambda x: model.module.decode(x).cpu()
                    generate_fid_samples(fn, epoch, opt.n_samples, opt.n_hidden, opt.fid_path_samples, device=device)
                    fid = get_fid(opt.fid_path_samples, opt.fid_path_pretrained)
                print('====> Epoch: {} Average loss: {:.4f} FID: {}'.format(
                    epoch, avg_loss, fid))

                # Log results
                logger.log({
                    "Epoch": epoch, 
                    "Avg Loss": avg_loss, 
                    "FID": fid
                    })

    tmp_epoch = 0
    for m in opt.load_path:
        epoch = load_model(m)

        # Quick fix to load multiple models and not have overwriting happening
        epoch = epoch if epoch is not tmp_epoch and tmp_epoch < epoch else tmp_epoch + 1
        tmp_epoch = epoch

        if opt.calc_fid:
            fn = lambda x: model.module.decode(x).cpu()
            generate_fid_samples(fn, epoch, opt.n_samples, opt.n_hidden, opt.fid_path_samples, device=device)
            fid = get_fid(opt.fid_path_samples, opt.fid_path_pretrained)
        if opt.test_recons:
            fn = lambda x: model(x.to(device))[0]
            gen_reconstructions(fn, test_loader, epoch, opt.test_results_path_recons, nrow=1, path_for_originals=opt.test_results_path_originals)
            print("Generated reconstructions")
        if opt.test_samples:
            fn = lambda x: model.module.decode(x).cpu()
            generate_samples(fn, epoch, 5, opt.n_hidden, opt.test_results_path_samples, nrow=1, device=device)
            print("Generated samples")


