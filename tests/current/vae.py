from __future__ import print_function
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



parser.add_argument('--dataset', type=str, default="celebA")
parser.add_argument('--image_root', type=str, default="./data")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--n_samples", type=int, default=10)
parser.add_argument('--n_z', type=int, nargs='+', default=[256, 8, 8]) # n_z
parser.add_argument('--input_channels', type=int, default=3)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--w_kld', type=float, default=1)
parser.add_argument('--w_loss_g', type=float, default=0.01)
parser.add_argument('--w_loss_gd', type=float, default=1)


opt = parser.parse_args()
# opt.cuda = not opt.no_cuda and torch.cuda.is_available()

torch.manual_seed(opt.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, _ = get_data_loader(opt)

class VAE(nn.Module):
    def __init__(self, opt, representation_size=64):
        super(VAE, self).__init__()

        # ENCODER 
        self.input_channels = opt.input_channels
        self.n_hidden = opt.n_hidden
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

        self.x_to_mu = nn.Sequential(
            nn.Linear(representation_size*4*8*8, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.n_hidden))
        
        self.x_to_logvar = nn.Sequential(
            nn.Linear(representation_size*4*8*8, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.n_hidden))


        # DECODER 
        self.input_size = opt.n_hidden
        self.representation_size2 = opt.n_z

        dim = self.representation_size2[0] * self.representation_size2[1] * self.representation_size2[2]
        self.preprocess = nn.Sequential(
            nn.Linear(self.input_size, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU())
            # 256 x 8 x 8
        self.deconv1 = nn.ConvTranspose2d(self.representation_size2[0], 256, 5, stride=2, padding=2)
        self.act1 = nn.Sequential(nn.BatchNorm2d(256),
                                  nn.ReLU())
            # 256 x 16 x 16
        self.deconv2 = nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2)
        self.act2 = nn.Sequential(nn.BatchNorm2d(128),
                                  nn.ReLU())
            # 128 x 32 x 32
        self.deconv3 = nn.ConvTranspose2d(128, 32, 5, stride=2, padding=2)
        self.act3 = nn.Sequential(nn.BatchNorm2d(32),
                                  nn.ReLU())
            # 32 x 64 x 64
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, stride=1, padding=2)
            # 3 x 64 x 64
        self.activation = nn.Tanh()


    def encode(self, x):
        batch_size = x.size()[0]
        inner = self.features(x).squeeze()
        inner = inner.view(batch_size, -1)
        mu = self.x_to_mu(inner)
        logvar = self.x_to_logvar(inner)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, code):
        bs = code.size()[0]
        print(code.size())
        sys.exit()
        preprocessed_codes = self.preprocess(code)
        preprocessed_codes = preprocessed_codes.view(-1,
                                                     self.representation_size2[0],
                                                     self.representation_size2[1],
                                                     self.representation_size2[2])
        output = self.deconv1(preprocessed_codes, output_size=(bs, 256, 16, 16))
        output = self.act1(output)
        output = self.deconv2(output, output_size=(bs, 128, 32, 32))
        output = self.act2(output)
        output = self.deconv3(output, output_size=(bs, 32, 64, 64))
        output = self.act3(output)
        output = self.deconv4(output, output_size=(bs, 3, 64, 64))
        output = self.activation(output)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

model = VAE(opt=opt)
model = torch.nn.DataParallel(model)
model = model.to(device)
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


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

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


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
#                                       recon_batch[:n]])
#                 save_image(comparison.cpu(),
#                          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in tqdm(range(1, opt.epochs + 1)):
        train(epoch)
        # test(epoch)
        with torch.no_grad():
            sample = torch.randn(1, opt.n_hidden).to(device)
            sample = model.module.decode(sample).cpu()
            save_image(sample.cpu(),
                       'results/sample_' + str(epoch) + '.png')
