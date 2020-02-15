import torch as T
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from collections import OrderedDict

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class Encoder_birds(nn.Module):
    def __init__(self, opt):
        super(Encoder_birds, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.avgpool = nn.AvgPool2d(4,1,0)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.x_to_mu = nn.Linear(512,opt.n_z)
        self.x_to_logvar = nn.Linear(512, opt.n_z)

    def reparameterize(self, x):
        mu = self.x_to_mu(x)
        logvar = self.x_to_logvar(x)
        z = T.randn(mu.size())
        z = get_cuda(z)
        z = mu + z * T.exp(0.5 * logvar)
        kld = (-0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))
        return z, kld

    def forward(self, x):
        x = self.resnet(x).squeeze()
        z, kld = self.reparameterize(x)
        return z, kld


class Generator_birds(nn.Module):
    def __init__(self, opt):
        super(Generator_birds, self).__init__()
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(opt.n_z, 512, 4, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 384, 4, 2, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, 4, 2, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        x_gen = self.convs(z)
        return x_gen


class Discriminator_birds(nn.Module):
    def __init__(self):
        super(Discriminator_birds, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):

        f_d = self.convs(x)
        x = self.last_conv(f_d)
        f_d = F.avg_pool2d(f_d, 4, 1, 0)
        return x.squeeze(), f_d.squeeze()


class Encoder_mnist(nn.Module):
    def __init__(self, opt):
        super(Encoder_mnist, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1f', nn.Conv2d(1, 16, 3, padding=1)),
            ('bn1f', nn.BatchNorm2d(16)),
            ('relu1f', nn.ReLU()),
            ('pool1f', nn.MaxPool2d(2, 2))
            ]))       

        self.mean =  nn.Sequential(OrderedDict([
            ('conv2m', nn.Conv2d(16, 4, 3, padding=1)),  
            ('bn2m', nn.BatchNorm2d(4)),
            ('relu2m', nn.ReLU()), 
            ('pool1m', nn.MaxPool2d(2, 2))
            ]))

        self.logvar =  nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(16, 4, 3, padding=1)),  
            ('bn2', nn.BatchNorm2d(4)),
            ('relu2', nn.ReLU()), 
            ('pool1', nn.MaxPool2d(2, 2))
            ]))

    def reparameterize(self, x):
        mu = self.mean(x).flatten()
        logvar = self.logvar(x).flatten()
        z = T.randn(mu.size())
        z = get_cuda(z)
        z = mu + z * T.exp(0.5 * logvar)
        kld = (-0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp(), 0))
        return z, kld

    def forward(self, x):
        x = self.features(x)
        z, kld = self.reparameterize(x)
        return z, kld


class Generator_mnist(nn.Module):
    def __init__(self, opt):
        super(Generator_mnist, self).__init__()

        self.decoder = nn.Sequential(OrderedDict([
            ('deconv1', nn.ConvTranspose2d(4, 16, 2, stride=2)),
            ('relu1', nn.ReLU()),
            ('deconv2', nn.ConvTranspose2d(16, 1, 2, stride=2)),
            ('sigmoid', nn.Sigmoid())
            ]))

    def forward(self, z):
        return self.decoder(z)


class Discriminator_mnist(nn.Module):
    def __init__(self):
        super(Discriminator_mnist, self).__init__()
        
        self.main = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1)),
             ('bn1', nn.BatchNorm2d(6)),
             ('relu1', nn.ReLU()),
             ('conv2', nn.Conv2d(in_channels=6,out_channels=12, kernel_size=3, stride=1)),
             ('bn2', nn.BatchNorm2d(12)),
             ('relu2', nn.ReLU())
             ]))    

        self.lth_features = nn.Sequential(nn.Linear(6912, 1024),
                nn.ReLU() )

        self.validity = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, x):

        main = self.main(x)
        view = main.view(x.shape[0], -1)
        f_d = self.lth_features(view)
        x = self.validity(f_d)

        return x, f_d


class Encoder_mnist_test(nn.Module):
    def __init__(self, opt):
        super(Encoder_mnist_test, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.resnet.avgpool = nn.AvgPool2d(4,1,0)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.x_to_mu = nn.Linear(512,opt.n_z)
        self.x_to_logvar = nn.Linear(512, opt.n_z)

    def reparameterize(self, x):
        mu = self.x_to_mu(x)
        logvar = self.x_to_logvar(x)
        z = T.randn(mu.size())
        z = get_cuda(z)
        z = mu + z * T.exp(0.5 * logvar)
        kld = (-0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))
        return z, kld

    def forward(self, x):
        x = self.resnet(x).squeeze()
        z, kld = self.reparameterize(x)
        return z, kld


class Generator_mnist_test(nn.Module):
    def __init__(self, opt):
        super(Generator_mnist_test, self).__init__()
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(opt.n_z, 512, 4, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 384, 4, 2, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, 4, 2, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        x_gen = self.convs(z)
        return x_gen


class Discriminator_mnist_test(nn.Module):
    def __init__(self):
        super(Discriminator_mnist_test, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):

        f_d = self.convs(x)
        x = self.last_conv(f_d)
        f_d = F.avg_pool2d(f_d, 4, 1, 0)
        return x.squeeze(), f_d.squeeze()


class Encoder_celeba(nn.Module):
    def __init__(self, opt):
        super(Encoder_celeba, self).__init__()

        # TODO
        self.x_to_mu = None
        self.x_to_logvar = None

    def reparameterize(self, x):
        mu = self.x_to_mu(x)
        logvar = self.x_to_logvar(x)
        z = T.randn(mu.size())
        z = get_cuda(z)
        z = mu + z * T.exp(0.5 * logvar)
        kld = (-0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))
        return z, kld

    def forward(self, x):
        x = self.resnet(x).squeeze()
        z, kld = self.reparameterize(x)
        return z, kld


class Generator_celeba(nn.Module):
    def __init__(self, opt):
        super(Generator_celeba, self).__init__()

        # TODO
        self.convs = nn.Sequential(
        )


    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        x_gen = self.convs(z)
        return x_gen


class Discriminator_celeba(nn.Module):
    def __init__(self):
        super(Discriminator_celeba, self).__init__()

        # TODO
        self.convs = nn.Sequential(
        )

        # TODO
        self.last_conv = nn.Sequential(
        )

    def forward(self, x):

        f_d = self.convs(x)
        x = self.last_conv(f_d)
        f_d = F.avg_pool2d(f_d, 4, 1, 0)
        return x.squeeze(), f_d.squeeze()
