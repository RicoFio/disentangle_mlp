import torch
from torch import optim
from torch.nn import DataParallel

from dataset import *
from model import *
from tqdm import tqdm

from fid import get_fid
from logger import Logger
from envsetter import EnvSetter
from helper_functions import *

opt = EnvSetter("vaegan_baseline").get_parser()

train_loader, val_loader, test_loader = get_data_loader(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netEG = VAE(opt=opt)
netEG = netEG.to(device)
netEG = torch.nn.DataParallel(netEG)
netD = Discriminator_celeba(opt).to(device)

optimizerEG = optim.Adam(netEG.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)

def load_model(path):
    checkpoint = torch.load(path)
    netEG.module.load_state_dict(checkpoint['encoder_decoder_model'])
    netD.load_state_dict(checkpoint['discriminator_model'])
    optimizerEG.load_state_dict(checkpoint['encoder_decoder_optimizer'])
    optimizerD.load_state_dict(checkpoint['discriminator_optimizer'])
    return checkpoint['epoch']

if __name__ == "__main__":
    tmp_epoch = 0
    for m in opt.load_path:
        epoch = load_model(m)

        # Quick fix to load multiple models and not have overwriting happening
        epoch = epoch if epoch is not tmp_epoch and tmp_epoch < epoch else tmp_epoch + 1
        tmp_epoch = epoch

        if opt.calc_fid:
            fn = lambda x: netEG.module.decode(x).cpu()
            generate_fid_samples(fn, epoch, opt.n_samples, opt.n_hidden, opt.fid_path_recons, device=device)
            fid = get_fid(opt.fid_path_recons, opt.fid_path_pretrained)
        if opt.test_recons:
            fn = lambda x: netEG(x.to(device))[0]
            gen_reconstructions(fn, test_loader, epoch, opt.test_results_path_recons, nrow=1, path_for_originals=opt.test_results_path_originals)
            print("Generated reconstructions")
        if opt.test_samples:
            fn = lambda x: netEG.module.decode(x).cpu()
            generate_samples(fn, epoch, 5, opt.n_hidden, opt.test_results_path_samples, nrow=1, device=device)
            print("Generated samples")
