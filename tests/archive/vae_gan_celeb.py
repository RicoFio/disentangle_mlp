#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import argparse
import h5py
import numpy as np
import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import torchvision.utils as vutils


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

def show_and_save(file_name,img):
    npimg = np.transpose(img.numpy(),(1,2,0))
    f = "./%s.png" % file_name
    fig = plt.figure(dpi=300)
    fig.suptitle(file_name, fontsize=14, fontweight='bold')
    #plt.imshow(npimg)
    plt.imsave(f,npimg)
    
def save_model(epoch, encoder, decoder, D):
    torch.save(decoder.cpu().state_dict(), './VAE_GAN_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(),'./VAE_GAN_encoder_%d.pth' % epoch)
    torch.save(D.cpu().state_dict(), 'VAE_GAN_D_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()
    D.cuda()
    
def load_model(epoch, encoder, decoder, D):
    #  restore models
    decoder.load_state_dict(torch.load('./VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load('./VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
    D.load_state_dict(torch.load('VAE_GAN_D_%d.pth' % epoch))
    D.cuda()


# In[ ]:


class CelebADataset(Dataset):
    def __init__(self, h5_path, transform=None):
        assert (os.path.isfile(h5_path))
        self.h5_path = h5_path
        self.transform = transform
        
        # loading the dataset into memory
        f = h5py.File(self.h5_path, "r")
        key = list(f.keys())
        print ("key list:", key)
        self.dataset = f[key[0]]
        self.dataset = self.dataset[:1000]
        print ("dataset loaded and its shape:", self.dataset.shape)
    
    def __getitem__(self, index):
        img = self.dataset[index]
        img = np.transpose(img, (1, 2, 0))
        if self.transform is not None:
            img = self.transform(img)
            
        return img, 0
    
    def __len__(self):
        return len(self.dataset)


# In[ ]:


batch_size = 64
T = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
train_loader = torch.utils.data.DataLoader(
    CelebADataset('../dataset/CelebA_aligned_reduced.h5', transform=T),
    batch_size=batch_size, shuffle=True)


# In[ ]:


data, _ = next(iter(train_loader))
print (data.size())
print (data[1, :])
show_and_save("gen" ,make_grid((data*0.5+0.5).cpu(),8))


# In[ ]:


class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, representation_size = 64):
        super(Encoder, self).__init__()
        # input parameters
        self.input_channels = input_channels
        self.output_channels = output_channels
        
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
    
    def hidden_layer(self, x):
        batch_size = x.size()[0]
        output = self.features(x)
        return output


# In[ ]:


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
        
            # 256 x 8 x 8
        self.deconv1 = nn.ConvTranspose2d(representation_size[0], 256, 5, stride=2, padding=2)
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


# In[ ]:


class VAE_GAN_Generator(nn.Module):
    def __init__(self, input_channels, hidden_size, representation_size=(256, 8, 8)):
        super(VAE_GAN_Generator, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.representation_size = representation_size
        
        self.encoder = Encoder(input_channels, hidden_size)
        self.decoder = Decoder(hidden_size, representation_size)
        
    def forward(self, x):
        batch_size = x.size()[0]
        mean, logvar = self.encoder(x)
        std = logvar.mul(0.5).exp_()
        
        reparametrized_noise = Variable(torch.randn((batch_size, self.hidden_size))).cuda()

        reparametrized_noise = mean + std * reparametrized_noise

        rec_images = self.decoder(reparametrized_noise)
        
        return mean, logvar, rec_images


# In[ ]:


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
        batch_size = x.size()[0]
        features = self.main(x)
        lth_rep = self.lth_features(features.view(batch_size, -1))
        output = self.sigmoid_output(lth_rep)
        return output
    
    def similarity(self, x):
        batch_size = x.size()[0]
        features = self.main(x)
        lth_rep = self.lth_features(features.view(batch_size, -1))
        return lth_rep


# In[ ]:


# define constant
input_channels = 3
hidden_size = 64
max_epochs = 250
lr = 3e-4

beta = 5
alpha = 0.1
gamma = 15


# In[ ]:


G = VAE_GAN_Generator(input_channels, hidden_size).cuda()
D = Discriminator(input_channels).cuda()

criterion = nn.BCELoss()
criterion.cuda()

opt_enc = optim.RMSprop(G.encoder.parameters(), lr=lr)
opt_dec = optim.RMSprop(G.decoder.parameters(), lr=lr)
opt_dis = optim.RMSprop(D.parameters(), lr=lr * alpha)


# In[ ]:


fixed_noise = Variable(torch.randn(batch_size, hidden_size)).cuda()
data, _ = next(iter(train_loader))
fixed_batch = Variable(data).cuda()


# In[ ]:


for epoch in range(max_epochs):
    D_real_list, D_rec_enc_list, D_rec_noise_list, D_list = [], [], [], []
    g_loss_list, rec_loss_list, prior_loss_list = [], [], []
    for data, _ in train_loader:
        batch_size = data.size()[0]
        ones_label = Variable(torch.ones(batch_size)).cuda()
        zeros_label = Variable(torch.zeros(batch_size)).cuda()
        
        #print (data.size())
        datav = Variable(data).cuda()
        mean, logvar, rec_enc = G(datav)
        #print ("The size of rec_enc:", rec_enc.size())
        
        noisev = Variable(torch.randn(batch_size, hidden_size)).cuda()
        rec_noise = G.decoder(noisev)
        
        # train discriminator
        output = D(datav)
        errD_real = criterion(output, ones_label)
        D_real_list.append(output.data.mean())
        output = D(rec_enc)
        errD_rec_enc = criterion(output, zeros_label)
        D_rec_enc_list.append(output.data.mean())
        output = D(rec_noise)
        errD_rec_noise = criterion(output, zeros_label)
        D_rec_noise_list.append(output.data.mean())
        
        dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
        #print ("print (dis_img_loss)", dis_img_loss)
        D_list.append(dis_img_loss.data.mean())
        opt_dis.zero_grad()
        dis_img_loss.backward(retain_graph=True)
        opt_dis.step()
        
        # train decoder
        output = D(datav)
        errD_real = criterion(output, ones_label)
        output = D(rec_enc)
        errD_rec_enc = criterion(output, zeros_label)
        output = D(rec_noise)
        errD_rec_noise = criterion(output, zeros_label)
        
        similarity_rec_enc = D.similarity(rec_enc)
        similarity_data = D.similarity(datav)
        
        dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
        #print (dis_img_loss)
        gen_img_loss = - dis_img_loss
        
        g_loss_list.append(gen_img_loss.data.mean())
        rec_loss = ((similarity_rec_enc - similarity_data) ** 2).mean()
        rec_loss_list.append(rec_loss.data.mean())
        err_dec = gamma * rec_loss + gen_img_loss
        
        opt_dec.zero_grad()
        err_dec.backward(retain_graph=True)
        opt_dec.step()
        
        # train encoder
        prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
        prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mean.data)
        #print (prior_loss, mean, std)
        prior_loss_list.append(prior_loss.data.mean())
        err_enc = prior_loss + beta * rec_loss
        
        opt_enc.zero_grad()
        err_enc.backward()
        opt_enc.step()
        
    #save_model(epoch, G.encoder, G.decoder, D)
    _, _, rec_imgs = G(fixed_batch)
    show_and_save('rec_epoch_%d.png' % epoch ,make_grid((rec_imgs.data*0.5+0.5).cpu(),8))
    '''
    vutils.save_image(rec_imgs.data,
            'rec_epoch_%d.png' % epoch,
            normalize=True)
    '''
    samples = G.decoder(fixed_noise)
    vutils.save_image(samples.data,
            'sample_epoch_%d.png' % epoch,
            normalize=True)
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    print ('[%d/%d]: D_real:%.4f, D_enc:%.4f, D_noise:%.4f, Loss_D:%.4f, Loss_G:%.4f, rec_loss:%.4f, prior_loss:%.4f' 
           % (epoch, 
              max_epochs, 
              np.mean(D_real_list), 
              np.mean(D_rec_enc_list), 
              np.mean(D_rec_noise_list), 
              np.mean(D_list), 
              np.mean(g_loss_list),
              np.mean(rec_loss_list),
              np.mean(prior_loss_list)))
    

