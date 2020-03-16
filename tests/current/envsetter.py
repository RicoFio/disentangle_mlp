import argparse
import os
from pathlib import Path
import warnings

class EnvSetter():
    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description=description)
        self._parse()
        self._set_up_dirs()

    def _parse(self):
        self.parser.add_argument('--name', type=str, required=True)

        self.parser.add_argument('--seed', type=int, default=999,
                                help='random seed (default: 999)')
        self.parser.add_argument("--num_workers", type=int, default=4)
        self.parser.add_argument('--log_interval', type=int, default=10,
                                help='how many batches to wait before logging training status')
        self.parser.add_argument("--use_gpus", type=str, default="0,1")

        self.parser.add_argument('--load_path', type=str, nargs="+", default=[])
        self.parser.add_argument('--save_path', type=str, default=f"./data/%")
        self.parser.add_argument('--log_path', type=str, default=f"./data/%/log")
        self.parser.add_argument('--fid_path_pretrained', type=str, default="/home/shared/evaluation/fid/fid_stats_celeba.npz")

        self.parser.add_argument('--dataset', type=str, default="celebA")
        self.parser.add_argument('--image_root_train', type=str, default=f"/home/shared/data/%/train")
        self.parser.add_argument('--image_root_val', type=str, default=f"/home/shared/data/%/val")
        self.parser.add_argument('--image_root_test', type=str, default=f"/home/shared/data/%/test")

        self.parser.add_argument('--epochs', type=int, default=30, metavar='N',
                                help='number of epochs to train (default: 30)')
        self.parser.add_argument('--batch_size_train', type=int, default=256, metavar='N', 
                                help='input batch size for training (default: 256)')
        self.parser.add_argument('--batch_size_val', type=int, default=256, metavar='N', 
                                help='input batch size for validation (default: 128)')
        self.parser.add_argument('--batch_size_test', type=int, default=256, metavar='N', 
                                help='input batch size for testing (default: 128)')
        self.parser.add_argument("--n_samples", type=int, default=1000)
        self.parser.add_argument('--n_z', type=int, nargs='+', default=[256, 8, 8])
        self.parser.add_argument('--n_hidden', type=int, default=128)
        self.parser.add_argument('--lr', type=float, default=3e-4)
        self.parser.add_argument('--beta', type=float, default=50)

        self.parser.add_argument('--input_channels', type=int, default=3)
        self.parser.add_argument('--img_size', type=int, default=64)

        def str2bool(v):
            return True if v.lower() == 'true' else False

        self.parser.add_argument('--calc_fid', type=str2bool, default=True)
        self.parser.add_argument('--to_train', type=str2bool, default=True)

        self.parser = self.parser.parse_args()

        # Fix some entries
        self.parser.save_path = self.parser.save_path.replace('%', self.parser.name)
        self.parser.log_path = self.parser.log_path.replace('%', self.parser.name)
        self.parser.image_root_train = self.parser.image_root_train.replace('%', self.parser.dataset)
        self.parser.image_root_val = self.parser.image_root_val.replace('%', self.parser.dataset)
        self.parser.image_root_test = self.parser.image_root_test.replace('%', self.parser.dataset)

        

    def _set_up_dirs(self):
        save_path = self.parser.save_path

        path = Path(save_path)
        if path.exists():
            warnings.warn("Path exists and containing files could be overwritten", UserWarning)
            abr = input("Continue [y]/n \n") or 'y'
            if abr is not 'y':
                raise ValueError("User interrupted path creation")

        path.mkdir(parents=True, exist_ok=True)
        Path(save_path + '/models').mkdir(parents=True, exist_ok=True)
        Path(save_path + '/results').mkdir(parents=True, exist_ok=True)
        Path(save_path + '/results/recons').mkdir(parents=True, exist_ok=True)
        Path(save_path + '/results/samples').mkdir(parents=True, exist_ok=True)
        Path(save_path + '/test_results/recons').mkdir(parents=True, exist_ok=True)
        Path(save_path + '/test_results/samples').mkdir(parents=True, exist_ok=True)
        Path(save_path + '/fid_results/recons').mkdir(parents=True, exist_ok=True)
        Path(save_path + '/fid_results/samples').mkdir(parents=True, exist_ok=True)
        Path(self.parser.log_path).mkdir(parents=True, exist_ok=True)

        self.parser.model_path = save_path + '/models'
        self.parser.results_path_recons = save_path + '/results/recons'
        self.parser.results_path_samples = save_path + '/results/samples'
        self.parser.test_results_path_recons = save_path + '/test_results/recons'
        self.parser.test_results_path_samples = save_path + '/test_results/samples'
        self.parser.fid_path_recons = save_path + '/fid_results/recons'
        self.parser.fid_path_samples = save_path + '/fid_results/samples'

    def get_parser(self):
        return self.parser


class VAEsetter(EnvSetter):
    def __init__(self, description, folder):
        super().__init__(description, folder)

    def _parse(self):
        super()._parse()


class GANsetter(EnvSetter):
    def __init__(self, description, folder):
        super().__init__(description, folder)
        self._parse()

    def _parse(self):
        super()._parse()


class GAEVANsetter(VAEsetter, GANsetter):
    def __init__(self, description, folder):
        super().__init__(description, folder)


class BGAEVANsetter(GAEVANsetter):
    def __init__(self, description, folder):
        super().__init__(description, folder)

# Testing
if __name__=="__main__":
    name = 'test'
    p = EnvSetter(name).get_parser()
    save_path = p.save_path

    # Test 1
    assert Path(save_path + '/models').exists()
    assert Path(save_path + '/results').exists()
    assert Path(save_path + '/results/recons').exists()
    assert Path(save_path + '/results/samples').exists()
    assert Path(save_path + '/test_results/recons').exists()
    assert Path(save_path + '/test_results/samples').exists()
    assert Path(save_path + '/fid_results/recons').exists()
    assert Path(save_path + '/fid_results/samples').exists()

    # Test 2
    p = EnvSetter(name)
