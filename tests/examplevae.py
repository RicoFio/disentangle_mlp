from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import sys


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         layers = []
        
#         layers.append(nn.Linear(in_features=28*28, out_features=512, bias=True))
#         layers.append(nn.LeakyReLU(0.2, inplace=True))
        
#         layers.append(nn.Linear(in_features=512, out_features=256, bias=True))
#         layers.append(nn.LeakyReLU(0.2, inplace=True))
        
#         layers.append(nn.Linear(in_features=256, out_features=1, bias=True))
#         layers.append(nn.Sigmoid())

#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         validity = self.model(x)
#         return validity

#     def forward_l(self, x, l):

#         x = x.view(x.size(0), -1)
#         a = self.model[0](x)
#         b = self.model[1](a)
#         c = self.model[2](b)
#         d = self.model[3](c)
#         return d


# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()

#         self.fc1 = nn.Linear(784, 400)
#         self.fc21 = nn.Linear(400, 20)
#         self.fc22 = nn.Linear(400, 20)
#         self.fc3 = nn.Linear(20, 400)
#         self.fc4 = nn.Linear(400, 784)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 784))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar, z

# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()

#         self.fc1 = nn.Linear(784, 400)
#         self.fc21 = nn.Linear(400, 20)
#         self.fc22 = nn.Linear(400, 20)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         #TODO Check this formula
#         kld = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))
#         return mu + eps*std, kld

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 784))
#         z, kld = self.reparameterize(mu, logvar)
#         return z, mu, logvar, kld



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