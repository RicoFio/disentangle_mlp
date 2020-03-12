import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

def arg_parse():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
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
    parser.add_argument('--load_model', type=str, default="")

    def str2bool(v):
        if v.lower() == 'true':
            return True
        else:
            return False

    parser.add_argument('--fid', type=str2bool, default=False)
    parser.add_argument('--to_train', type=str2bool, default=True)

    return parser.parse_args()

save_path = "./data/vae/models/model_%.tar"

opt = arg_parse()

torch.manual_seed(opt.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, _ = get_data_loader(opt)

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


if __name__ == "__main__":
    if opt.load_model:
        checkpoint = torch.load(opt.load_model)
        model.load_state_dict(checkpoint['VAE_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

    if opt.to_train:
        # tempoaray change
        for epoch in tqdm(range(30)):
            train(epoch)
            with torch.no_grad():
                sample = torch.randn(10, opt.n_hidden).to(device)
                sample = model.module.decode(sample).cpu()
                save_image(sample.cpu(),
                        './data/vae/results/sample_' + str(epoch) + '.png')
                batch = next(iter(train_loader))
                batch = model.module.decode(model.module.encode(sample).cpu()).cpu()
                save_image(batch.cpu(),
                                './data/vae/quick_results/recon_' + str(epoch) + '.png')
                
	        if os.path.isfile(save_path.replace('%',epoch-4)):
		    os.remove(save_path.replace('%',epoch-4))
                torch.save({
                'epoch': epoch + 1,
                "VAE_model": model.module.state_dict(),
                'optimizer': optimizer.state_dict()}, save_path.replace('%',str(epoch+1)))
    elif opt.load_model and not opt.fid:
        sample = torch.randn(80, opt.n_hidden).to(device)
        sample = model.module.decode(sample).cpu()
        save_image(sample.cpu(),
                        './data/vae/quick_results/sample_' + str(epoch) + '.png')
        batch = next(iter(train_loader))
        batch = model.module.decode(model.module.encode(sample).cpu()).cpu()
        save_image(batch.cpu(),
                        './data/vae/quick_results/recon_' + str(epoch) + '.png')
    elif opt.load_model and opt.fid:
        sample = torch.randn(5000, opt.n_hidden).to(device)
        sample = model.module.decode(sample).cpu()

        for s in sample:
            save_image(sample.cpu(),
                        './data/vae/fid_results/sample_' + str(epoch) + '.png')
