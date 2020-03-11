import argparse
import os
from pathlib import Path
import warnings

class EnvSetter():
    def __init__(self, description, folder):
        self.parser = argparse.ArgumentParser(description=description)
        self._parse()
        self._set_up_dirs()

    def _parse(self):
        self.parser.add_argument('--name', type=str, required=True)

        self.parser.add_argument('--seed', type=int, default=1,
                                help='random seed (default: 1)')
        self.parser.add_argument("--num_workers", type=int, default=4)
        self.parser.add_argument('--log_interval', type=int, default=10,
                                help='how many batches to wait before logging training status')

        self.parser.add_argument('--load_path', type=str, default="")
        self.parser.add_argument('--save_path', type=str, default=f"./data/{self.parser.save_path}")
        self.parser.add_argument('--log_path', type=str, default=f"./data/{self.parser.save_path}/log")
        self.parser.add_argument('--fid_path_pretrained', type=str, default="/home/shared/save_riccardo/fid/celeba/fid_stats_celeba.npz")

        self.parser.add_argument('--dataset', type=str, default="celebA")
        self.parser.add_argument('--image_root_train', type=str, default=f"/shared/data/{self.parser.dataset}/train")
        self.parser.add_argument('--image_root_val', type=str, default=f"./data/{self.parser.dataset}/val")
        self.parser.add_argument('--image_root_test', type=str, default=f"./data/{self.parser.dataset}/test")

        self.parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                                help='input batch size for training (default: 128)')
        self.parser.add_argument('--epochs', type=int, default=30, metavar='N',
                                help='number of epochs to train (default: 10)')
        self.parser.add_argument("--n_samples", type=int, default=10)
        self.parser.add_argument('--n_z', type=int, nargs='+', default=[256, 8, 8])
        self.parser.add_argument('--n_hidden', type=int, default=128)
        self.parser.add_argument('--lr', type=float, default=3e-4)

        self.parser.add_argument('--input_channels', type=int, default=3)
        self.parser.add_argument('--img_size', type=int, default=64)

        def str2bool(v):
            return True if v.lower() == 'true' else False

        self.parser.add_argument('--calc_fid', type=str2bool, default=True)
        self.parser.add_argument('--to_train', type=str2bool, default=True)

    def _set_up_dirs(self):
        save_path = self.parser.save_path

        path = Path(save_path)
        if path.exists():
            warnings.warn("Path exists and containing files could be overwritten", UserWarning)
            _ = input("Press any key to continue")

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

        self.parser.add_argument('--model_path', type=str, default=save_path + '/models')
        self.parser.add_argument('--results_path_recons', type=str, default=save_path + '/results/recons')
        self.parser.add_argument('--results_path_samples', type=str, default=save_path + '/results/samples')
        self.parser.add_argument('--test_results_path_recons', type=str, default=save_path + '/test_results/recons')
        self.parser.add_argument('--test_results_path_samples', type=str, default=save_path + '/test_results/samples')
        self.parser.add_argument('--fid_path_recons', type=str, default=save_path + 'fid_results/recons')
        self.parser.add_argument('--fid_path_samples', type=str, default=save_path + 'fid_results/samples')

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