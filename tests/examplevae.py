from comet_ml import Experiment

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

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
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=3e-4, metavar='L',
                    help='learning rate')
args = parser.parse_args()

# Set seed
torch.manual_seed(args.seed)

# Load Device
if torch.cuda.device_count() > 1 and use_gpu:
    device = torch.cuda.current_device()
    enc.to(device)
    enc = nn.DataParallel(module=enc)
    dec.to(device)
    dec = nn.DataParallel(module=dec)
    dis.to(device)
    dis = nn.DataParallel(module=dis)
    kwargs = {'num_workers': 1, 'pin_memory': True}
elif torch.cuda.device_count() == 1 and use_gpu:
    device =  torch.cuda.current_device()
    enc.to(device)
    dec.to(device)
    dis.to(device)
    kwargs = {'num_workers': 1, 'pin_memory': True}
else:
    device = torch.device('cpu')
    kwargs = {}

# Load Data - MNIST
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor()),
                    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=False,
                    transform=transforms.ToTensor()),
                    batch_size=args.batch_size, shuffle=True, **kwargs)
# Load Data - EMNIST
# print("Downloading Dataset")
# train_loader = DataLoader(
#     datasets.EMNIST('../data/gzip', train=True, download=True, split="balanced",
#                     transform=transforms.ToTensor()),
#                     batch_size=100, shuffle=True)
# test_loader = DataLoader(
#     datasets.EMNIST('../data/gzip', train=False, download=True, split="balanced",
#                     transform=transforms.ToTensor()),
#                     batch_size=100, shuffle=True)

# Set up experiment
experiment = Experiment(api_key="cTXulwAAXRQl33uirmViBlbK4",
                        project_name="mlp-cw3", workspace="ricofio")

######################################
#### Class Definitions
######################################
class Discriminator(nn.Module):
    def __init__(self, input_channels, representation_size=(256, 8, 8)):  
        super(Discriminator, self).__init__()
        self.representation_size = representation_size
        dim = representation_size[0] * representation_size[1] * representation_size[2]
        
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        
        self.lth_features = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.LeakyReLU(0.2))
        
        self.sigmoid_output = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid())
        
    def forward(self, x):
        '''Return lth feature and validity of fake'''
        batch_size = x.size()[0]
        features = self.main(x)
        lth_rep = self.lth_features(features.view(batch_size, -1))
        output = self.sigmoid_output(lth_rep)
        return lth_rep, output


class Encoder(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, representation_size = 64):
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

        self.mean = nn.Sequential(
            nn.Linear(representation_size*4*8*8, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_channels))
        
        self.logvar = nn.Sequential(
            nn.Linear(representation_size*4*8*8, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_channels))

    def forward(self, x):

        batch_size = x.size()[0]

        hidden_representation = self.features(x)

        mu = self.mean(hidden_representation.view(batch_size, -1))

        logvar = self.logvar(hidden_representation.view(batch_size, -1))
        
        # reparameterization 
        std = torch.exp(0.5*logvar)

        eps = torch.randn_like(std)

        z = mu + eps*std

        return z, mu, logvar




class Decoder(nn.Module):

    def __init__(self, input_size, representation_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.representation_size = representation_size
        dim = representation_size[0] * representation_size[1] * representation_size[2]
        
        self.preprocess = nn.Sequential(
            nn.Linear(input_size, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU())
        
        self.decode = nn.Sequential(
            # 256 x 8 x 8
            nn.ConvTranspose2d(representation_size[0], 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            # 256 x 16 x 16
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            # 128 x 32 x 32
            nn.ConvTranspose2d(128, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            # 32 x 64 x 64
            nn.ConvTranspose2d(32, 3, 5, stride=1, padding=2),
            # 3 x 64 x 64
            nn.Tanh())

    def forward(self, code):

        bs = code.size()[0]
        preprocessed_codes = self.preprocess(code)
        preprocessed_codes = preprocessed_codes.view(-1,
                                                     self.representation_size[0],
                                                     self.representation_size[1],
                                                     self.representation_size[2])
        output = self.deconv1(preprocessed_codes, output_size=(bs, 256, 16, 16))
        output = self.act1(output)
        output = self.deconv2(output, output_size=(bs, 128, 32, 32))
        output = self.act2(output)
        output = self.deconv3(output, output_size=(bs, 32, 64, 64))
        output = self.act3(output)
        output = self.deconv4(output, output_size=(bs, 3, 64, 64))
        output = self.activation(output)
        return output
        
######################################

# Define Models
#TODO Add weight_init
encoder = Encoder()
encoder.to_device(device)

decoder = Decoder()
decoder.to_device(device)

discriminator = Discriminator()
discriminator.to_device(device)

# Optimizers
optimizer_enc = optim.RMSprop(enc.parameters(), lr=lr)
optimizer_dec = optim.RMSprop(dec.parameters(), lr=lr)
optimizer_dis = optim.RMSprop(dis.parameters(), lr=lr)

# sys.exit()

######################################
#### Loss Helpers Definitions
######################################
def loss_llikelihood(discriminator, recon_x, x):

    recon_x_lth, _  = discriminator.forward(recon_x)

    x_lth, _ = discriminator.forward(x)

    res = F.mse_loss( recon_x_lth , x_lth , reduction= 'sum')

    return res

def loss_prior(recon_x, x, mu, logvar):

    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def loss_discriminator(discriminator, decoder, z, x):

    left = torch.log(discriminator.forward(x))

    right = torch.log( 1. - discriminator(decoder(z)) )

    res = left + right 

    return res

def loss_encoder(discriminator, recon_x, x, mu, logvar):

    return loss_prior(recon_x, x, mu, logvar) + loss_llikelihood(discriminator, recon_x, x)

def loss_decoder(discriminator, recon_x, x, decoder, z, gamma=0.5):

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

    with experiment.train():
        for batch_idx, (data, _) in tqdm(enumerate(train_loader)):

            data = data.to(device)

            # encoder
            # TODO Adapt to neew classes 
            optimizer_enc.zero_grad()   

            # encoder takes data and returns 
            z, mu, logvar = encoder(data)

            loss_enc = torch.mean(loss_encoder(discriminator, recon_batch, data, mu, logvar))
            loss_enc.backward(retain_graph=True)
            optimizer_enc.step()

            # decoder 
            recon_batch = decoder(z)

            optimizer_dec.zero_grad()
            loss_dec = torch.mean(loss_decoder(discriminator, recon_batch, data, decoder, z, gamma=0.01))
            loss_dec.backward(retain_graph=True)
            optimizer_dec.step()

            # discriminator
            optimizer_dis.zero_grad()
            # TODO this line is prob wrong gonna have to check
            loss_dis = torch.mean(loss_discriminator(discriminator, decoder, z, recon_batch))
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
            sample = torch.randn(args.batch_size, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(args.batch_size, 1, args.representation_size, args.representation_size),
                       'results/sample_' + str(epoch) + '.png')
