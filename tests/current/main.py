import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from model import *
import random
from helper_functions import *
from dataset import get_data_loader
from dataset import *
import torchvision.utils as utils
import argparse

from matplotlib import pyplot as plt

import torchvision.models
from torchvision.models.resnet import model_urls

#model_urls['resnet18'] = model_urls['resnet18'].replace('https://', 'http://')

from tqdm import tqdm

save_path = "./data/results/model.tar"
# load_path = "home/shared/saved_models/celeba_models/saved_model_epoch_73.tar"

if not os.path.exists("data/saved_models"):
    os.makedirs("data/saved_models")

# Hyperparameters for lfw
## Epoch size 250
## Batch size 64
## Img size 64
## ? Crop size 150
## Recon VS gan weight 1e-6
## Real VS gan weight 0.33
## discriminate ae recon False
## Discriminate sample z True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="celebA")
parser.add_argument('--image_root', type=str, default="./data")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=101)
parser.add_argument('--lr_e', type=float, default=0.0003)
parser.add_argument('--lr_g', type=float, default=0.0003)
parser.add_argument('--lr_d', type=float, default=0.0003)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--n_samples", type=int, default=36)
parser.add_argument('--n_z', type=int, nargs='+', default=[256, 8, 8])
parser.add_argument('--input_channels', type=int, default=3)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--w_kld', type=float, default=1)
parser.add_argument('--w_loss_g', type=float, default=0.01)
parser.add_argument('--w_loss_gd', type=float, default=1)

def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False

parser.add_argument('--resume_training', type=str2bool, default=False)
parser.add_argument('--to_train', type=str2bool, default=True)

opt = parser.parse_args()
print(opt)

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
T.manual_seed(manual_seed)

if T.cuda.is_available():
    T.cuda.manual_seed_all(manual_seed)

train_loader, _ = get_data_loader(opt)
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

if opt.dataset == "birds":
    E = Encoder_birds(opt).to(device)
    G = Generator_birds(opt).to(device).apply(weights_init)
    D = Discriminator_birds(opt).to(device).apply(weights_init)

elif opt.dataset == "mnist":
    E = Encoder_mnist_test(opt).to(device)
    G = Generator_mnist_test(opt).to(device).apply(weights_init)
    D = Discriminator_mnist_test(opt).to(device).apply(weights_init)

elif opt.dataset == "celebA":
    E = Encoder_celeba(opt).to(device)
    G = Generator_celeba(opt).to(device).apply(weights_init)
    D = Discriminator_celeba(opt).to(device).apply(weights_init)


E.to(device)
G.to(device)
D.to(device)

E = nn.DataParallel(E)
G = nn.DataParallel(G)
D = nn.DataParallel(D)


E_trainer = T.optim.Adam(E.parameters(), lr=opt.lr_e)
G_trainer = T.optim.Adam(G.parameters(), lr=opt.lr_g, betas=(0.5, 0.999))
D_trainer = T.optim.Adam(D.parameters(), lr=opt.lr_d, betas=(0.5, 0.999))

def train_batch(x_r):
    batch_size = x_r.size(0)
    y_real = T.ones(batch_size).to(device)
    y_fake = T.zeros(batch_size).to(device)

    #Extract latent_z corresponding to real images
    z, kld = E(x_r)
    kld = kld.mean()
    #Extract fake images corresponding to real images
    x_f = G(z)

    #Extract latent_z corresponding to noise
    z_p = T.randn(batch_size, opt.n_hidden)
    z_p = z_p.to(device)
    #Extract fake images corresponding to noise
    x_p = G(z_p)

    #Compute D(x) for real and fake images along with their features
    ld_r, fd_r = D(x_r)
    ld_f, fd_f = D(x_f)
    ld_p, fd_p = D(x_p)

    #------------D training------------------
    loss_D = F.binary_cross_entropy(ld_r, y_real) + 0.5*(F.binary_cross_entropy(ld_f, y_fake) + F.binary_cross_entropy(ld_p, y_fake))
    D_trainer.zero_grad()
    loss_D.backward(retain_graph = True)
    D_trainer.step()

    #------------E & G training--------------

    #loss corresponding to -log(D(G(z_p)))
    loss_GD = F.binary_cross_entropy(ld_p, y_real)
    #pixel wise matching loss and discriminator's feature matching loss
    loss_G = 0.5 * (0.01*(x_f - x_r).pow(2).sum() + (fd_f - fd_r.detach()).pow(2).sum()) / batch_size

    E_trainer.zero_grad()
    G_trainer.zero_grad()
    (opt.w_kld*kld+opt.w_loss_g*loss_G+opt.w_loss_gd*loss_GD).backward()
    E_trainer.step()
    G_trainer.step()


    return loss_D.item(), loss_G.item(), loss_GD.item(), kld.item()

def load_model_from_checkpoint():
    global E, G, D, E_trainer, G_trainer, D_trainer
    checkpoint = T.load(save_path)
    E.load_state_dict(checkpoint['E_model'])
    G.load_state_dict(checkpoint['G_model'])
    D.load_state_dict(checkpoint['D_model'])
    E_trainer.load_state_dict(checkpoint['E_trainer'])
    G_trainer.load_state_dict(checkpoint['G_trainer'])
    D_trainer.load_state_dict(checkpoint['D_trainer'])
    return checkpoint['epoch']

def training():
    start_epoch = 0
    if opt.resume_training:
        start_epoch = load_model_from_checkpoint()

    for epoch in tqdm(range(start_epoch, opt.epochs)):
        E.train()
        G.train()
        D.train()

        T_loss_D = []
        T_loss_G = []
        T_loss_GD = []
        T_loss_kld = []

        for x, _ in tqdm(train_loader):
            x = x.to(device)
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

        generate_samples(opt, "data/results/%d.jpg" % epoch)
        T.save({
            'epoch': epoch + 1,
            "E_model": E.state_dict(),
            "G_model": G.state_dict(),
            "D_model": D.state_dict(),
            'E_trainer': E_trainer.state_dict(),
            'G_trainer': G_trainer.state_dict(),
            'D_trainer': D_trainer.state_dict()
        }, save_path)




# THIS MAY NOT WORK 
# def train_batch(x_r):
#     batch_size = x_r.size(0)
#     y_real = T.ones(batch_size).to(device)
#     y_fake = T.zeros(batch_size).to(device)

#     #Extract latent_z corresponding to real images
#     z, kld = E(x_r)
#     kld = kld.mean()


#     ## Train with all-real batch
#     D_trainer.zero_grad()
#     ld_r, fd_r = D(x_r)
#     loss_D_real = F.binary_cross_entropy(ld_r, y_real)
#     loss_D_real.backward(retain_graph=True)
#     print(loss_D_real)

#     ## Train with all-fake batch

#     #Extract fake images corresponding to real images
#     x_f = G(z).to(device)

#     #Extract latent_z corresponding to noise
#     z_p = T.randn(batch_size, opt.n_hidden).to(device)
#     #Extract fake images corresponding to noise
#     x_p = G(z_p).to(device)

#     #Compute D(x) for real and noise 
#     ld_f, fd_f = D(x_f.detach())
#     ld_p, fd_p = D(x_p.detach())

#     loss_D_fake = (F.binary_cross_entropy(ld_f, y_fake) + F.binary_cross_entropy(ld_p, y_fake))

#     loss_D_fake.backward(retain_graph=True)
#     print(loss_D_fake)
#     loss_D = loss_D_real + loss_D_fake

#     #------------D training------------------
#     # loss_D = F.binary_cross_entropy(ld_r, y_real) + 0.5*(F.binary_cross_entropy(ld_f, y_fake) + F.binary_cross_entropy(ld_p, y_fake))
#     # loss_D.backward(retain_graph = True)

#     D_trainer.step()

#     #------------E & G training--------------

#     #loss corresponding to -log(D(G(z_p)))
#     loss_GD = F.binary_cross_entropy(ld_p, y_real)
#     #pixel wise matching loss and discriminator's feature matching loss
#     loss_G = 0.5 * ((x_f - x_r).pow(2).sum() + (fd_f - fd_r.detach()).pow(2).sum()) / batch_size

#     E_trainer.zero_grad()
#     G_trainer.zero_grad()
#     total_loss =  (opt.w_kld*kld+opt.w_loss_g*loss_G+opt.w_loss_gd*loss_GD)
#     print(total_loss, "total loss")
#     total_loss.backward()
#     E_trainer.step()
#     G_trainer.step()


#     return loss_D.item(), loss_G.item(), loss_GD.item(), kld.item()

# def load_model_from_checkpoint():
#     global E, G, D, E_trainer, G_trainer, D_trainer
#     checkpoint = T.load(save_path)
#     E.load_state_dict(checkpoint['E_model'])
#     G.load_state_dict(checkpoint['G_model'])
#     D.load_state_dict(checkpoint['D_model'])
#     E_trainer.load_state_dict(checkpoint['E_trainer'])
#     G_trainer.load_state_dict(checkpoint['G_trainer'])
#     D_trainer.load_state_dict(checkpoint['D_trainer'])
#     return checkpoint['epoch']

# def training():
#     start_epoch = 0
#     if opt.resume_training:
#         start_epoch = load_model_from_checkpoint()

#     for epoch in tqdm(range(start_epoch, opt.epochs)):
#         E.train()
#         G.train()
#         D.train()

#         T_loss_D = []
#         T_loss_G = []
#         T_loss_GD = []
#         T_loss_kld = []

#         for x, _ in tqdm(train_loader):
#             #plt.imshow(np.transpose(utils.make_grid(x[0][:64], padding=2, normalize=True).cpu(),(1,2,0)))
#             #plt.show()
#             x = x.to(device)
#             loss_D, loss_G, loss_GD, loss_kld = train_batch(x)
#             T_loss_D.append(loss_D)
#             T_loss_G.append(loss_G)
#             T_loss_GD.append(loss_GD)
#             T_loss_kld.append(loss_kld)


#         T_loss_D = np.mean(T_loss_D)
#         T_loss_G = np.mean(T_loss_G)
#         T_loss_GD = np.mean(T_loss_GD)
#         T_loss_kld = np.mean(T_loss_kld)

#         print("epoch:", epoch, "loss_D:", "%.4f"%T_loss_D, "loss_G:", "%.4f"%T_loss_G, "loss_GD:", "%.4f"%T_loss_GD, "loss_kld:", "%.4f"%T_loss_kld)

#         for sample in range(opt.n_samples):
#             generate_samples("data/results/ " , str(epoch) ,  "-" , str(sample) ,".jpg" )


#         T.save({
#             'epoch': epoch + 1,
#             "E_model": E.state_dict(),
#             "G_model": G.state_dict(),
#             "D_model": D.state_dict(),
#             'E_trainer': E_trainer.state_dict(),
#             'G_trainer': G_trainer.state_dict(),
#             'D_trainer': D_trainer.state_dict()
#         }, save_path)


def generate_samples(opt, img_name):
    z_p = T.randn(opt.n_samples, opt.n_hidden)
    z_p = z_p.to(device)
    E.eval()
    G.eval()
    D.eval()
    with T.autograd.no_grad():
        x_p = G(z_p)
    utils.save_image(x_p.cpu(), img_name, normalize=True)



if __name__ == "__main__":
    if opt.to_train:
        training()
    else:
        checkpoint = T.load(save_path)
        G.load_state_dict(checkpoint['G_model'])
        generate_samples(opt, "data/testing_img.jpg")
