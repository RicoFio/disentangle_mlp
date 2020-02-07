import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataloader.dataloader import EMNISTPytorchDataProvider

from comet_ml import Experiment

import os

# Command line arguments
args = get_args()
# Set seeds
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(seed=args.seed)

# Import data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# Load data
train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

if args.block_type == 'vae_gan':
    processing_block_type = None
    dim_reduction_block_type = None
else:
    raise ModuleNotFoundError

custom_conv_net = ConvolutionalNetwork(  # initialize our network object, in this case a ConvNet
    input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_width),
    num_output_classes=args.num_classes, num_filters=args.num_filters, use_bias=False,
    num_blocks_per_stage=args.num_blocks_per_stage, num_stages=args.num_stages,
    processing_block_type=processing_block_type,
    dimensionality_reduction_block_type=dim_reduction_block_type)

conv_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data_loader, val_data=val_data_loader,
                                    test_data=test_data_loader,
                                    optimizer=args.lr_rule,
                                    lr=args.lr)  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics