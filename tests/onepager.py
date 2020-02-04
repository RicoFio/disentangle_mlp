from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torchvision.utils as utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
import os
import numpy as np
import time

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(
        # nc x 64 x 64
        nn.Conv2d(self.input_channels, representation_size, 5, stride=2, padding=2),
        nn.BatchNorm2d(representation_size),
        nn.ReLU(),
        # hidden_size x 32 x 32
        nn.Conv2d(representation_size, representation_size*2, 5, stride=2, padding=2),
        nn.BatchNorm2d(representation_size * 2),
        nn.ReLU(),
        # hidden_size*2 x 16 x 16
        nn.Conv2d(representation_size*2, representation_size*4, 5, stride=2, padding=2),
        nn.BatchNorm2d(representation_size * 4),
        nn.ReLU())
        # hidden_size*4 x 8 x 8

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #TODO Check this formula
        kld = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))
        return mu + eps*std, kld

    def forward(self, x):

        mu, logvar = self.encode(x.view(-1, 784))
        z, kld = self.reparameterize(mu, logvar)
        return z, mu, logvar, kld

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc3 = nn.Linear(28, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        l_layers = []
        l_layers.append(nn.Linear(  in_features=28*28, out_features=512, bias=True))
        l_layers.append(nn.LeakyReLU(0.2, inplace=True))
        l_layers.append(nn.Linear(in_features=512, out_features=256, bias=True))
        l_layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.l_layers = nn.Sequential(*l_layers)

        self.last_layer = nn.Sequential(
        nn.Linear(in_features=256, out_features=1, bias=True),
        nn.Sigmoid())

    def forward(self, x):
        f_d = self.l_layers(x)
        validity = self.last_layer(f_d)
        return validity.squeeze(), f_d.squeeze()

class VaeGanLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(VaeGanLoss, self).__init__(size_average, reduce, reduction)

    def loss_llikelihood(self, discriminator, recon_x, x):
        return F.mse_loss( discriminator.forward_l(recon_x) , discriminator.forward_l(x) , reduction= 'sum')

    def loss_prior(self, recon_x, x, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def loss_discriminator(self, discriminator, decoder, z, x):
        left = torch.log(discriminator.forward(x))
        right = torch.log( 1. - discriminator(decoder.forward(z)) )
        return left + right 

    def loss_encoder(self, discriminator, recon_x, x, mu, logvar):
        return self.loss_prior(recon_x, x, mu, logvar) + self.loss_llikelihood(discriminator, recon_x, x)

    def loss_decoder(self, discriminator, recon_x, x, decoder, z, gamma=0.5):
        return gamma * self.loss_llikelihood(discriminator, recon_x, x) - self.loss_discriminator(discriminator, decoder, z, x)


# Define Models
enc = Encoder()
dec = Decoder()
dis = Discriminator()

# Load Device
if torch.cuda.device_count() > 1 and use_gpu:
    device = torch.cuda.current_device()
    enc.to(device)
    enc = nn.DataParallel(module=enc)
    dec.to(device)
    dec = nn.DataParallel(module=dec)
    dis.to(device)
    dis = nn.DataParallel(module=dis)
elif torch.cuda.device_count() == 1 and use_gpu:
    device =  torch.cuda.current_device()
    enc.to(device)
    dec.to(device)
    dis.to(device)
else:
    device = torch.device('cpu')

# Initialize weights
# dnc.apply()
# dis.apply()

# Load Data
print("Downloading Dataset")
train_loader = DataLoader(
    datasets.EMNIST('./data/gzip', train=True, download=True, split="balanced",
                    transform=transforms.ToTensor()),
                    batch_size=100, shuffle=True)
test_loader = DataLoader(
    datasets.EMNIST('./data/gzip', train=False, download=True, split="balanced",
                    transform=transforms.ToTensor()),
                    batch_size=100, shuffle=True)

# Create Experiment
experiment = Experiment(api_key="cTXulwAAXRQl33uirmViBlbK4",
                        project_name="mlp-cw3", workspace="ricofio")

# Define Optimizers
lr = 3e-4
optimizer_enc = optim.RMSprop(enc.parameters(), lr=lr)
optimizer_dec = optim.RMSprop(dec.parameters(), lr=lr)
optimizer_dis = optim.RMSprop(dis.parameters(), lr=lr)

def train_batch(x_r):
    batch_size = x_r.size(0)
    y_real = torch.ones(batch_size)
    y_fake = torch.zeros(batch_size)

    #Extract latent_z and fake image
    z, mu, logvar, kld = enc(x_r)
    x_f = dec(x_r)
    #Extract latent_z corresponding to noise; 100 is the size of the latent variable here
    z_p = torch.randn(batch_size, 100)
    #Extract fake images corresponding to noise
    x_p = dec(z_p)

    #Compute D(x) for real and fake images along with their features
    ld_r, fd_r = dis(x_r)
    ld_f, fd_f = dis(x_f)
    ld_p, fd_p = dis(x_p)

    #------------Dis training------------------
    loss_D = F.binary_cross_entropy(ld_r, y_real) + 0.5*(F.binary_cross_entropy(ld_f, y_fake) + F.binary_cross_entropy(ld_p, y_fake))
    optimizer_dis.zero_grad()
    loss_D.backward(retain_graph = True)
    optimizer_dis.step()

    #------------Enc & Dec training--------------

    #loss corresponding to -log(D(G(z_p)))
    loss_GD = F.binary_cross_entropy(ld_p, y_real)
    #pixel wise matching loss and discriminator's feature matching loss
    loss_G = 0.5 * (0.01*(x_f - x_r).pow(2).sum() + (fd_f - fd_r.detach()).pow(2).sum()) / batch_size

    optimizer_enc.zero_grad()
    optimizer_dec.zero_grad()
    w_loss_g = 0.05
    w_loss_gd = 0.05
    (kld+w_loss_g*loss_G+w_loss_gd*loss_GD).backward()
    optimizer_enc.step()
    optimizer_dec.step()

    return loss_D.item(), loss_G.item(), loss_GD.item(), kld.item()


def training():
    start_epoch = 0
    epochs = 100
    with experiment.train():

        for epoch in tqdm(range(start_epoch, epochs)):
            enc.train()
            dec.train()
            dis.train()

            T_loss_D = []
            T_loss_G = []
            T_loss_GD = []
            T_loss_kld = []

            for x, _ in tqdm(train_loader):
                loss_D, loss_G, loss_GD, loss_kld = train_batch(x)
                T_loss_D.append(loss_D)
                T_loss_G.append(loss_G)
                T_loss_GD.append(loss_GD)
                T_loss_kld.append(loss_kld)


            T_loss_D = np.mean(T_loss_D)
            T_loss_G = np.mean(T_loss_G)
            T_loss_GD = np.mean(T_loss_GD)
            T_loss_kld = np.mean(T_loss_kld)

            print("epoch:", epoch, "loss_D:", "%.4f"%T_loss_D, "loss_G:", "%.4f"%T_loss_G, "loss_GD:", "%.4f"%T_loss_GD, "loss_kld:", "%.4f"%T_loss_kld)

            generate_samples(f"data/results/{epoch}.jpg")


def generate_samples(img_name):
    z_p = torch.randn(5, 100)
    enc.eval()
    dec.eval()
    dis.eval()
    
    with torch.autograd.no_grad():
        x_p = dec(z_p)
    utils.save_image(x_p.cpu(), img_name, normalize=True, nrow=6)



if __name__ == "__main__":
    training()
    generate_samples("data/testing_img.jpg")