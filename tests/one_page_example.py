
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from collections import OrderedDict

from tqdm import tqdm

import os
import sys

# Parse command line argumetns
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--channel-nr', type=int, default=1, metavar='N',
                    help='input number of channel for data (default: 1)')
parser.add_argument('--representation-size', type=int, default=28, metavar='N',
                    help='input pixel size of data elements (default: 28)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=3e-4, metavar='L',
                    help='learning rate')
args = parser.parse_args()

######################################
#### Class Definitions
######################################

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)),
            ('bn1', nn.BatchNorm2d(16)),
            ('relu1', nn.ReLU()),
            ( 'pool1', nn.MaxPool2d(2, 2))
            ]))       

        self.mean =  nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(16, 4, 3, padding=1)),  
            ('bn2', nn.BatchNorm2d(4)),
            ('relu2', nn.ReLU()), 
            ( 'pool1', nn.MaxPool2d(2, 2))
            ]))

        self.logvar =  nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(16, 4, 3, padding=1)),  
            ('bn2', nn.BatchNorm2d(4)),
            ('relu2', nn.ReLU()), 
            ( 'pool1', nn.MaxPool2d(2, 2))
            ]))

    def forward(self, x):
        hidden_representation = self.features(x)
        
        mu = self.mean(hidden_representation)
        logvar = self.logvar(hidden_representation)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(OrderedDict([
            ('convt1', nn.ConvTranspose2d(4, 16, 2, stride=2)),
            ('relu1', nn.ReLU()), 
            ('convt3', nn.ConvTranspose2d(16, 1, 2, stride=2)),
            ('convt4', nn.ConvTranspose2d(1, 1, 2, stride=1)),
            ('convt5', nn.ConvTranspose2d(1, 1, 2, stride=1)),
            ('convt6', nn.ConvTranspose2d(1, 1, 2, stride=1)),
            ('convt7', nn.ConvTranspose2d(1, 1, 2, stride=1)),
            ]))

    def forward(self, z):
        return torch.sigmoid(self.decoder(z))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1)),
             ('bn1', nn.BatchNorm2d(6)),
             ('relu1', nn.ReLU()),
             ('conv2', nn.Conv2d(in_channels=6,out_channels=12, kernel_size=3, stride=1)),
             ('bn2', nn.BatchNorm2d(12)),
             ('relu2', nn.ReLU())
             ]))    

        self.lth_features = nn.Sequential( nn.Linear(6912, 1024),
                nn.ReLU() )

        self.validity = nn.Sequential( nn.Linear(1024, 1), nn.Sigmoid() )

    def forward(self, x):

        main = self.main(x)
        lth_features = self.lth_features(main.view(128, -1))

        return lth_features, self.validity(lth_features)
# Set seed
torch.manual_seed(args.seed)


# Define Models
#TODO Add weight_init
encoder = Encoder()
# encoder.to_device(device)

decoder = Decoder()
# decoder.to_device(device)

discriminator = Discriminator()
# discriminator.to_device(device)

# Load Device
if torch.cuda.device_count() > 1:
    device = torch.cuda.current_device()
    encoder.to(device)
    encoder = nn.DataParallel(module=encoder)
    decoder.to(device)
    decoder = nn.DataParallel(module=decoder)
    discriminator.to(device)
    discriminator = nn.DataParallel(module=discriminator)
    kwargs = {'num_workers': 1, 'pin_memory': True}
elif torch.cuda.device_count() == 1:
    device =  torch.cuda.current_device()
    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)
    kwargs = {'num_workers': 1, 'pin_memory': True}
else:
    device = torch.device('cpu')
    kwargs = {}

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
            transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)



# Optimizers
optimizer_enc = optim.RMSprop(encoder.parameters(), lr=args.lr)
optimizer_dec = optim.RMSprop(decoder.parameters(), lr=args.lr)
optimizer_dis = optim.RMSprop(discriminator.parameters(), lr=args.lr)

# Great Global Variables - GGVs
ones = torch.ones((128,1)).to(device)
zeros = torch.zeros((128,1)).to(device)

######################################
#### Loss Helpers Definitions
######################################
def loss_llikelihood(discriminator, recon_x, x):

    recon_x_lth, _  = discriminator.forward(recon_x)

    x_lth, _ = discriminator.forward(x)

    res = F.mse_loss( recon_x_lth , x_lth , reduction= 'mean')

    return res

def loss_prior(recon_x, x, mu, logvar):

    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def loss_discriminator(discriminator, decoder, z, x):
    _, res_1 = discriminator(x)
    _, res_2 = discriminator(decoder(z))

    real_loss = F.binary_cross_entropy(res_1, ones)
    fake_loss = F.binary_cross_entropy(res_2, zeros)

    # _ , i = discriminator.forward(x)
    # left = torch.log(i)

    # _ , j = discriminator(decoder(z))
    # right = torch.log(1. - j)

    # res = left + right 

    # return res

    return real_loss + fake_loss

def loss_encoder(discriminator, recon_x, x, mu, logvar):

    return loss_prior(recon_x, x, mu, logvar) + loss_llikelihood(discriminator, recon_x, x)

def loss_decoder(discriminator, recon_x, x, decoder, z, gamma=1):

    return gamma * loss_llikelihood(discriminator, recon_x, x) - loss_discriminator(discriminator, decoder, z, x)
######################################

def train(epoch):

    # Set models into train mode
    encoder.train()
    decoder.train()
    discriminator.train()

    train_loss_dec = 0
    train_loss_enc = 0
    train_loss_dis = 0


    for batch_idx, (data, _) in tqdm(enumerate(train_loader)):

        data = data.to(device)

            # encoder
        optimizer_enc.zero_grad()   

            # encoder takes data and returns 
        z, mu, logvar = encoder(data)
        recon_batch = decoder(z)

        loss_enc = torch.mean(loss_encoder(discriminator, recon_batch, data, mu, logvar))
        loss_enc.backward(retain_graph=True)
        optimizer_enc.step()

            # decoder 
        optimizer_dec.zero_grad()
        loss_dec = torch.mean(loss_decoder(discriminator, recon_batch, data, decoder, z, gamma=0.01))
        loss_dec.backward(retain_graph=True)
        optimizer_dec.step()

            # discriminator
        optimizer_dis.zero_grad()
            
        loss_dis = torch.mean(loss_discriminator(discriminator, decoder, z, data))
        loss_dis.backward()
        optimizer_dis.step()

            # Summed error over epoch
        train_loss_dec += loss_dec.item()
        train_loss_enc += loss_enc.item()
        train_loss_dis += loss_dis.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss dec: {:.6f} \t  Loss enc: {:.6f} \t Loss dis: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_dec.item() / len(data), 
                loss_enc.item() / len(data),
                loss_dis.item() / len(data),))
        # Temporary early stopping
        if batch_idx == 30:
            break

    print('====> Epoch: {} Average decoder loss: {:.4f}'.format(
          epoch, train_loss_dec / len(train_loader.dataset)))

# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in tqdm(range(1, args.epochs + 1)):
        train(epoch)
        # test(epoch)
        with torch.no_grad():
            sample = torch.randn((args.batch_size, 4, 6, 6)).to(device)
            sample = decoder.forward(sample).to(device)
            save_image(sample.view(args.batch_size, 1, args.representation_size, args.representation_size),
                    'results/sample_' + str(epoch) + '.png')
