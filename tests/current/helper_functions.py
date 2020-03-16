import torch
from torchvision.utils import save_image
import warnings
from pathlib import Path

def gen_fid_reconstructions(fn, dl, epoch, results_path):
    with torch.no_grad():
        orig_imgs, _ = next(iter(dl))
        batch = fn(orig_imgs).cpu()
        for i,x in enumerate(batch):
            save_image(x.cpu(), results_path + f'/recon_{i}_{str(epoch)}.pdf', normalize=True)

def gen_reconstructions(fn, dl, epoch, results_path, nrow=8, path_for_originals=""):
    with torch.no_grad():
        orig_imgs, _ = next(iter(dl))
        batch = fn(orig_imgs).cpu()
        save_image(batch.cpu(), results_path + f'/recon_{str(epoch)}.pdf',nrow=nrow, normalize=True)
        if path_for_originals:
            save_image(orig_imgs.cpu(), path_for_originals + f'/original_{str(epoch)}.pdf', normalize=True)

def generate_fid_samples(fn, epoch, n_samples, n_hidden, results_path, device="cpu"):
    with torch.no_grad():
        sample = torch.randn(n_samples, n_hidden).to(device)
        sample = fn(sample).cpu()
        for i, x in enumerate(sample):
            save_image(x.cpu(), results_path + f'/sample_{i}_{str(epoch)}.pdf', normalize=True)

def generate_samples(fn, epoch, n_samples, n_hidden, results_path, nrow=8, device="cpu"):
    with torch.no_grad():
        sample = torch.randn(n_samples, n_hidden).to(device)
        sample = fn(sample).cpu()
        save_image(sample.cpu(), results_path + f'/sample_{str(epoch)}.pdf', nrow=nrow, normalize=True)

# Testing
if __name__=="__main__":
    n_samples = 1
    n_hidden = 64
    test_data = torch.zeros(1,3,64,64)
    dl=[(test_data,1)]
    fn = lambda x: x
    fn1 = lambda x: torch.zeros(n_samples,3,64,64)
    path = r"C:\Users\ricca\Desktop\mlp\tests\current\test_log"

    # Tests
    gen_fid_reconstructions(fn, dl, 0, path)
    assert Path(path + r'\recon_0_0.pdf').exists()
    gen_reconstructions(fn, dl, 1, path)
    assert Path(path + r'\recon_1.pdf').exists()
    gen_reconstructions(fn, dl, 2, path , True, path)
    assert Path(path + r'\recon_2.pdf').exists()
    assert Path(path + r'\original_2.pdf').exists()
    generate_fid_samples(fn1, 3, 1, 64, path)
    assert Path(path + r'\sample_0_3.pdf').exists()
    generate_samples(fn1, 4, 1, 64, path)
    assert Path(path + r'\sample_4.pdf').exists()

    print("All tests passed")
