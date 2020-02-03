
import torch 
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

class VaeGanLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(VaeGanLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, discriminator, decoder, z, recon_x, x, mu, logvar):
        
        return self.vae_gan_loss(discriminator, decoder, z, recon_x, x, mu, logvar)


	def loss_llikelihood(self, discriminator, recon_x, x):

		return F.mse_loss( discriminator.forward_l(recon_x) , discriminator.forward_l(x) , reduction= 'sum')


	def loss_prior(self, recon_x, x, mu, logvar):

		return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


	def loss_gan(self, discriminator, decoder, z, x):

		left = torch.log(discriminator.forward(x))

		right = torch.log( 1. - discriminator(decoder.forward(z)) )

		return left + right 


	def vae_gan_loss(self, discriminator, decoder, z, recon_x, x, mu, logvar):

		return self.loss_prior(recon_x, x, mu, logvar) + self.loss_llikelihood(discriminator, recon_x, x) + self.loss_gan(discriminator, decoder, z, x)
