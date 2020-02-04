from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import sys


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

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

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


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
        mean = self.mean(hidden_representation.view(batch_size, -1))
        logvar = self.logvar(hidden_representation.view(batch_size, -1))
        return mean, logvar


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
            
    def reparametrize(self, x, mean, logvar):
        batch_size = x.size()[0]
        std = logvar.mul(0.5).exp_()
        
        reparametrized_noise = torch.randn((batch_size, self.hidden_size))

        reparametrized_noise = mean + std * reparametrized_noise

        rec_images = self.decoder(reparametrized_noise)
        
        return mean, logvar, rec_images

    def forward(self, code):
        reparametrized_noise = self.reparametrize(code)
        preprocessed_codes = self.preprocess(reparametrized_noise)
        preprocessed_codes = preprocessed_codes.view(-1,
                                                     self.representation_size[0],
                                                     self.representation_size[1],
                                                     self.representation_size[2])
        return self.decode(preprocessed_codes)



model = VAE().to(device)
discriminator = Discriminator()
optimizer_enc = optim.Adam(params= list(model.fc1.parameters()) + list(model.fc21.parameters()) + list(model.fc22.parameters()) , lr=1e-3)
optimizer_dec = optim.Adam(params= list(model.fc3.parameters()) + list(model.fc4.parameters()) , lr=1e-3)
optimizer_dis = optim.Adam(params= discriminator.parameters(), lr=1e-3)


def loss_llikelihood(discriminator, recon_x, x):

    res = F.mse_loss( discriminator.forward_l(recon_x, l=1) , discriminator.forward_l(x, l=1) , reduction= 'sum')
    # print(res)
    # sys.exit()
    return res

def loss_prior(recon_x, x, mu, logvar):

    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def loss_discriminator(discriminator, vae, z, x):

    left = torch.log(discriminator.forward(x))

    right = torch.log( 1. - discriminator(vae.decode(z)) )

    res = left + right 

    return res

def loss_encoder(discriminator, recon_x, x, mu, logvar):

    return loss_prior(recon_x, x, mu, logvar) + loss_llikelihood(discriminator, recon_x, x)

def loss_decoder(discriminator, recon_x, x, vae, z, gamma=0.5):

    return gamma * loss_llikelihood(discriminator, recon_x, x) - loss_discriminator(discriminator, vae, z, x)

def train(epoch):

    model.train()
    # discriminator.train()

    train_loss_dec = 0
    for batch_idx, (data, _) in enumerate(train_loader):

        data = data.to(device)

        # order 

        # encoder 
        optimizer_enc.zero_grad()

        # forward pass
        recon_batch, mu, logvar, z = model(data)

        loss_enc = torch.mean(loss_encoder(discriminator, recon_batch, data, mu, logvar))
        loss_enc.backward(requires=)
        optimizer_enc.step()

        # decoder 
        optimizer_dec.zero_grad()
        loss_dec = torch.mean(loss_decoder(discriminator, recon_batch, data, model, z, gamma=0.01))
        loss_dec.backward()
        optimizer_dec.step()

        # discriminator
        optimizer_dis.zero_grad()
        loss_dis = torch.mean(loss_discriminator(discriminator, model, z, recon_batch))
        loss_dis.backward()
        optimizer_dis.step()


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
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')