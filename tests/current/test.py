############################
# Imports
############################


import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import argparse
import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dataset import *
from model import *
from tqdm import tqdm

from datetime import datetime 

import json

from fid import get_fid

############################

# Globals

# function to add to JSON 
def write_json(data): 
    with open(log_path,'w') as f: 
        json.dump(data, f, indent=4) 
      
def log(results):
    with open(log_path) as json_file: 
        data = json.load(json_file) 
        
        temp = data['output'] 
        temp.append(results) 
        
    write_json(data)  

def arg_parse():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default="celebA")
    parser.add_argument('--image_root', type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument('--n_z', type=int, nargs='+', default=[256, 8, 8]) # n_z
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--load_model', type=str, default="")
    parser.add_argument('--save_path', type=str, default="./data/vaegan")
    parser.add_argument('--log_path', type=str, default="./data/vaegan/log")
    parser.add_argument('--fid_path_pretrained', type=str, default="/home/shared/save_riccardo/fid/celeba/fid_stats_celeba.npz")

    def str2bool(v):
        if v.lower() == 'true':
            return True
        else:
            return False

    parser.add_argument('--fid', type=str2bool, default=False)
    parser.add_argument('--to_train', type=str2bool, default=True)

    return parser.parse_args()



opt = arg_parse()


def set_up_log(path=opt.log_path):
    global log_path
    log_file = f"/log_{str(datetime.now())}.json"
    log_path= path + log_file

    args = vars(opt)
    empty_log = {
        "meta_data" : {
            "file": os.path.basename(__file__),
            "datetime": str(datetime.now()),
            "args": args
        },
        "output" : []
    }

    write_json(empty_log)
 
# Create necessary folder structure
def set_up_dirs():
    Path(opt.save_path).mkdir(parents=True, exist_ok=True)
    Path(opt.save_path + '/models').mkdir(parents=True, exist_ok=True)
    Path(opt.save_path + '/results').mkdir(parents=True, exist_ok=True)
    Path(opt.save_path + '/quick_results').mkdir(parents=True, exist_ok=True)
    Path(opt.save_path + '/fid_results').mkdir(parents=True, exist_ok=True)
    Path(opt.log_path).mkdir(parents=True, exist_ok=True)

log_path = ""
set_up_log()
save_path = opt.save_path + "/models/model_%.tar"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(opt.seed)

# Load data 
train_loader, test_loader = get_data_loader(opt)

model = VAE(opt=opt)
model = torch.nn.DataParallel(model)
model = model.to(device)
model.apply(weights_init)
       
netD = Discriminator_celeba(opt).to(device)
netD.apply(weights_init)

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)

# Initialize BCELoss function
criterion = nn.BCELoss()
#############################

#def set_up_globals():
#    global opt, model, netD, optimizer, optimizerD, criterion, train_loader, test_loader, log_path, save_path, device

#     opt = arg_parse()
#     set_up_log()
#     save_path = opt.save_path + "/models/model_%.tar"
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     torch.manual_seed(opt.seed)
# 
#     # Load data 
#     train_loader, test_loader = get_data_loader(opt)
# 
#     model = VAE(opt=opt)
#     model = torch.nn.DataParallel(model)
#     model = model.to(device)
#     model.apply(weights_init)
#        
#     netD = Discriminator_celeba(opt).to(device)
#     netD.apply(weights_init)
# 
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     optimizerD = optim.Adam(netD.parameters(), lr=1e-3)
# 
#     # Initialize BCELoss function
#     criterion = nn.BCELoss()



# Reconstruction + KL divergence losses summed over all elements and batch
def reconstruction_loss(recon_x, x, mu, logvar,  **kwargs):

    MSE = F.mse_loss(recon_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if is_gen:
        sim_real = kwargs['sim_real']
        sim_recon = kwargs['sim_recon']
        # then add similarity
        SIM = F.mse_loss(sim_recon, sim_real, reduction='sum')

        return MSE + SIM
    else:
        return MSE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in tqdm(enumerate(train_loader)):
        # create labels 
        fake_label = np.random.choice(a=[0.1,0.9], p=[0.95, 0.05])
        real_label = np.random.choice(a=[0.1,0.9], p=[0.05, 0.95])
        data = data.to(device)

        ### Discriminator ###

        netD.zero_grad()

        label = torch.full((data.size()[0],), real_label, device=device)
        # Forward pass real batch through D
        output, sim_real = netD(data)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(data.size()[0], 128, device=device)
        # Generate fake image batch with G
        fake = model.module.decode(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output, _ = netD(fake.detach())
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)

        # Calculate the gradients for this batch
        errD_fake.backward()
        # Update D
        optimizerD.step()

        ### Decoder ###

        model.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost

        # encoder to reuires grad = False
        model.module.features.requires_grad = False
        model.module.x_to_mu.requires_grad = False
        model.module.x_to_logvar.requires_grad = False
        model.module.preprocess.requires_grad = True
        model.module.deconv1.requires_grad = True
        model.module.act1.requires_grad = True
        model.module.deconv2.requires_grad = True
        model.module.act2.requires_grad = True
        model.module.deconv3.requires_grad = True
        model.module.act3.requires_grad = True
        model.module.deconv4.requires_grad = True
        model.module.activation.requires_grad = True       
        recon_batch, mu, logvar = model(data)

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output_fake, _ = netD(fake)

        # should add this too
        output_recon, sim_recon = netD(recon_batch)
       
        # Calculate G's loss based on this output
        errG_fake = criterion(output_fake, label)
        errG_recon = criterion(output_recon, label)
        # Calculate gradients for G
        errG_fake.backward()
        errG_recon.backward()
        loss = reconstruction_loss(recon_x=recon_batch.to(device), x=data, mu=mu.to(device), logvar=logvar.to(device), is_gen=True, sim_real=sim_real, sim_recon=sim_recon)
        loss.backward()
        optimizer.step()

        ### Encoder ###
        model.zero_grad()

        model.module.features.requires_grad = True
        model.module.x_to_mu.requires_grad = True
        model.module.x_to_logvar.requires_grad = True
        model.module.preprocess.requires_grad = False
        model.module.deconv1.requires_grad = False
        model.module.act1.requires_grad = False
        model.module.deconv2.requires_grad = False
        model.module.act2.requires_grad = False
        model.module.deconv3.requires_grad = False
        model.module.act3.requires_grad = False
        model.module.deconv4.requires_grad = False
        model.module.activation.requires_grad = False

        recon_batch, mu, logvar = model(data)

        loss = reconstruction_loss(recon_x=recon_batch.to(device), x=data, mu=mu.to(device), logvar=logvar.to(device), is_gen=False)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # Calculate FID score
    generate_samples(epoch, 100)
    fid = get_fid(opt.save_path + '/fid_results/', opt.fid_path_pretrained)

    # Log epoch statistics
    avg_loss = train_loss / len(train_loader.dataset)
    log({"Epoch":epoch, "Avg Loss":avg_loss, "FID":fid})
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} FID: {fid}')

def generate_reconstructions(epoch, results_path="results", singles=True, store_origs=True, fid=True):
    with torch.no_grad():
        orig_imgs, _ = next(iter(test_loader)) if fid else next(iter(train_loader))
        batch = model(orig_imgs)[0].cpu()
        if singles:
            for i,x in enumerate(batch):
                save_image(x.cpu(), opt.save_path + f'/{"fid_results" if fid else results_path}/recon_{i}_{str(epoch)}.png')
        else:
            save_image(batch.cpu(), opt.save_path + f'/{results_path}/recon_{str(epoch)}.png')

        if store_origs:
            save_image(orig_imgs.cpu(), opt.save_path + f'/originals/origin_{str(epoch)}.png')

def generate_samples(epoch, n_samples, results_path="results", singles=True, fid=True):
    with torch.no_grad():
        sample = torch.randn(n_samples, opt.n_hidden).to(device)
        sample = model.module.decode(sample).cpu()
        if singles:
            for i, x in enumerate(sample):
                save_image(x.cpu(), opt.save_path + f'/{"fid_results" if fid else results_path}/sample_{i}_{str(epoch)}.png')
        else:
            save_image(sample.cpu(), opt.save_path + f'/{results_path}/sample_{str(epoch)}.png')


if __name__ == "__main__":
#    set_up_globals()

    start_epoch = 0
    if opt.load_model:
        checkpoint = torch.load(opt.load_model, map_location="cuda:0")
        model.module.load_state_dict(checkpoint['encoder_decoder_model'])
        optimizer.load_state_dict(checkpoint['encoder_decoder_optimizer'])
        netD.load_state_dict(checkpoint['discriminator_model'])
        optimizerD.load_state_dict(checkpoint['discriminator_optimizer'])
        start_epoch = checkpoint['epoch']

    if opt.to_train:
        for epoch in tqdm(range(start_epoch, opt.epochs)):
            train(epoch)
            
            generate_reconstructions(epoch, singles=False, fid=False)
            generate_samples(epoch, 80, singles=False, fid=False)
            model_path = opt.save_path + "/models/model_%.tar"
            if os.path.isfile(model_path.replace('%', str(epoch-5))):
                os.remove(model.replace('%',str(epoch-5)))
            torch.save({
            'epoch': epoch + 1,
            "encoder_decoder_model": model.module.state_dict(),
            "discriminator_model": netD.state_dict(),
            'encoder_decoder_optimizer': optimizer.state_dict(),
            'discriminator_optimizer': optimizerD.state_dict(),
            }, save_path.replace('%',str(epoch+1)))
                           
    # Generate a cluster of images from reconstructions and samples
    elif opt.load_model and not opt.fid:
        generate_reconstructions(epoch, results_path="quick_results", singles=False, fid=False)
        generate_samples(epoch, n_samples=80,results_path="quick_results", singles=False)
    # Generate images for FID analysis
    elif opt.load_model and opt.fid:
        generate_reconstructions(epoch, results_path="fid_results")
        generate_samples(epoch, n_samples=1000,results_path="fid_results")
