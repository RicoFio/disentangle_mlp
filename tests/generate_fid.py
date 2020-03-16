import torch
from helper_functions import *
from envsetter import EnvSetter
from logger import Logger
from models import *
from torch import optim

opt = EnvSetter("fid_gen").get_parser()
logger = Logger(opt.log_path, opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VAE(opt=opt)
netD = Discriminator_celeba(opt).to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)

if __name__ == "__main__":
#    set_up_globals()
    
    checkpoint = torch.load(opt.load_model, map_location="cuda:0")
    model.load_state_dict(checkpoint['encoder_decoder_model'])
    optimizer.load_state_dict(checkpoint['encoder_decoder_optimizer'])
    netD.load_state_dict(checkpoint['discriminator_model'])
    optimizerD.load_state_dict(checkpoint['discriminator_optimizer'])
    epoch = checkpoint['epoch']

    with torch.no_grad():
        # Calculate FID
        fn = lambda x: netG(x).detach().cpu()
        generate_fid_samples(fn, epoch, opt.n_samples, opt.n_hidden, opt.fid_path_samples, device=device)
        fid = get_fid(opt.fid_path_samples, opt.fid_path_pretrained)
        # Log stats
        logger.log({
            "Epoch": epoch, 
            "Avg Loss G": "N/A", 
            "Avg Loss E": "N/A",
            "FID": fid
        })