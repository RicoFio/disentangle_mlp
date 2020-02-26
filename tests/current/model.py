import torch as T
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from collections import OrderedDict

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
        z = z
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
    def __init__(self, opt):
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
        z = z
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
    def __init__(self, opt):
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
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=2, padding=40,
                               bias=False)
        self.resnet.avgpool = nn.AvgPool2d(4,1,0)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.x_to_mu = nn.Linear(512,opt.n_z)
        self.x_to_logvar = nn.Linear(512, opt.n_z)

    def reparameterize(self, x):
        mu = self.x_to_mu(x) 
        logvar = self.x_to_logvar(x)
        z = T.randn(mu.size())
        z = z
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
        #self.convs = nn.Sequential(
        self.conv1 = nn.ConvTranspose2d(opt.n_z, 512, 7, 1, 0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(512, 384, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(384)
        self.conv3 = nn.ConvTranspose2d(384, 192, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(192)
        self.conv4 = nn.ConvTranspose2d(192, 1, 5, 1, 2, bias=False)
        self.tanh = nn.Tanh()
        #)


    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        x_gen = self.conv1(z)
        x_gen = self.relu(x_gen)
        x_gen = self.conv2(x_gen)
        x_gen = self.bn1(x_gen)
        x_gen = self.relu(x_gen)
        x_gen = self.conv3(x_gen)
        x_gen = self.bn2(x_gen)
        x_gen = self.relu(x_gen)
        x_gen = self.conv4(x_gen)
        x_gen = self.tanh(x_gen)

        return x_gen


class Discriminator_mnist_test(nn.Module):
    def __init__(self, opt):
        super(Discriminator_mnist_test, self).__init__()
        #self.convs = nn.Sequential(
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        #)

        self.last_conv = nn.Sequential(
            nn.Conv2d(256, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        f_d = self.conv1(x)
        print(f_d.shape, "\n")
        f_d = self.lrelu(f_d)
        f_d = self.conv2(f_d)
        print(f_d.shape, "\n")
        f_d = self.bn1(f_d)
        f_d = self.lrelu(f_d)
        f_d = self.conv3(f_d)
        print(f_d.shape, "\n")
        f_d = self.bn2(f_d)
        f_d = self.lrelu(f_d)
        x = self.last_conv(f_d)
        print(x.shape, "\n")
        f_d = F.avg_pool2d(f_d, 3, 1, 0)
        return x.squeeze(), f_d.squeeze()


class Encoder_celeba(nn.Module):
    def __init__(self, opt, representation_size=64):
        super(Encoder_celeba, self).__init__()

        self.input_channels = opt.input_channels
        self.output_channels = opt.output_channels
        
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
            
        self.x_to_mu = nn.Sequential(
            nn.Linear(representation_size*4*8*8, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.output_channels))
        
        self.x_to_logvar = nn.Sequential(
            nn.Linear(representation_size*4*8*8, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.output_channels))

    def reparameterize(self, x):
        mu = self.x_to_mu(x)
        logvar = self.x_to_logvar(x)
        z = T.randn(mu.size())
        z = z
        z = mu + z * T.exp(0.5 * logvar)
        kld = (-0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))
        return z, kld

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.features(x).squeeze()
        z, kld = self.reparameterize(x.view(batch_size, -1))
        return z, kld


class Generator_celeba(nn.Module):
    def __init__(self, opt):
        super(Generator_celeba, self).__init__()

        self.input_size = 64
        self.representation_size = opt.n_z

        dim = self.representation_size[0] * self.representation_size[1] * self.representation_size[2]

        self.preprocess = nn.Sequential(
            nn.Linear(self.input_size, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU())
        
            # 256 x 8 x 8
        self.deconv1 = nn.ConvTranspose2d(self.representation_size[0], 256, 5, stride=2, padding=2)
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


class Discriminator_celeba(nn.Module):
    def __init__(self, opt):
        super(Discriminator_celeba, self).__init__()

        self.representation_size = opt.n_z
        dim = self.representation_size[0] * self.representation_size[1] * self.representation_size[2]
        
        self.convs = nn.Sequential(
            nn.Conv2d(opt.input_channels, 32, 5, stride=1, padding=2),
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
        f_d = self.convs(x)
        x = self.lth_features(f_d.view(batch_size, -1))
        f_d = self.sigmoid_output(x)

        return f_d.squeeze(), x.squeeze()

class Test_Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Test_Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output