from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time

from models.loss import VaeGanLoss

from pytorch_mlp_framework.storage_utils import save_statistics
import wandb

class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data, weight_decay_coefficient, use_gpu, optimizer, lr, continue_from_epoch=-1):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = experiment_name
        self.model = network_model
        # TODO populate hyper_parameters
        self.hyper_parameters = {}

        # Find GPUs and transfer computation
        if torch.cuda.device_count() > 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
            print('Using Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device =  torch.cuda.current_device()
            self.model.to(self.device)
            print('Using GPU', self.device)
        else:
            print("Using CPU")
            self.device = torch.device('cpu')
            print(self.device)

        # Comet ML Setup
        self.experiment = Experiment(api_key="cTXulwAAXRQl33uirmViBlbK4",
                        project_name="mlp-cw3", workspace="ricofio")
                        
        # Re-initialize Model Weights
        self.model.reset_parameters()

        # Set data
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        # Optimizer setting ADD if need be
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), weight_decay=weight_decay_coefficient, lr=lr)
        else:
            self.optimizer = optim.Adam(self.parameters(), amsgrad=False,
                                        weight_decay=weight_decay_coefficient, lr=lr)

        # Set CosineAnnealingLR as learning_rate_scheduler
        self.learning_rate_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                            T_max=num_epochs,
                                                                            eta_min=0.00002)
        # Generate directory names for experiment
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        # Generate folders if not existing
        if not os.path.exists(self.experiment_folder):
            os.mkdir(self.experiment_folder)
            os.mkdir(self.experiment_logs)
            os.mkdir(self.experiment_saved_models)
        
        # For early stopping
        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.
        self.num_epochs = num_epochs

        # send loss computation to GPU
        self.criterion1 = VaeGanLoss().to(self.device)
        self.criterion2 = nn.CosineSimilarity().to(self.device)

        # Training starting point
        # If continue from epoch is -2 then continue from latest saved model
        if continue_from_epoch == -2:
            self.state, self.best_val_model_idx, self.best_val_model_acc = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx='latest')
            self.starting_epoch = int(self.state['model_epoch'])

        # If continue from epoch is greater than -1 then continue from existing model
        elif continue_from_epoch > -1:
            # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.state, self.best_val_model_idx, self.best_val_model_acc = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)
            self.starting_epoch = continue_from_epoch
        else:
            self.state = dict()
            self.starting_epoch = 0

        # Log the hyperparameters
        self.experiment.log_parameters(hyper_parameters)

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def run_train_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        # Set model to training mode
        self.train()
        # Send data to device as torch tensors
        x, y = x.float().to(device=self.device), y.long().to(device=self.device)
        # Forward data in model
        out = self.model.forward(x)

        # Compute loss  
        loss = VaeGanLoss(input=out, target=y)

        # Set all weight grads from previous training iters to 0
        self.optimizer.zero_grad()
        # Backpropagate to compute gradients for current iter loss
        loss.backward()

        self.learning_rate_scheduler.step(epoch=self.current_epoch)
        # Update network parameters
        self.optimizer.step()

        # Get argmax of predictions
        _, predicted = torch.max(out.data, 1)
        # Compute accuracy
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))
        return loss.cpu().data.numpy(), accuracy

    def run_validation_iter(self, x, y):
        """
        For validation run
        Receives the inputs and targets for the model and runs an evaluation iteration. 
        Returns loss and accuracy metrics.
        :param x: The inputs to the model
        :param y: The targets for the model
        :return:  The loss and accuracy for this batch
        """
        # Sets the system to validation mode
        self.eval()
        # Convert data to pytorch tensors and send to the computation device
        x, y = x.float().to(device=self.device), y.long().to(device=self.device)
        # Forward the data in the model  
        out = self.model.forward(x)

        # Compute loss
        loss = VaeGanLoss(input=out, target=y)

        # Get argmax of predictions
        _, predicted = torch.max(out.data, 1)
        # Compute accuracy
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))
        return loss.cpu().data.numpy(), accuracy

    def run_test_iter(self, x, y):
        """
        For validation run
        Receives the inputs and targets for the model and runs an evaluation iteration. 
        Returns loss and accuracy metrics.
        :param x: The inputs to the model
        :param y: The targets for the model
        :return:  The loss and accuracy for this batch
        """
        # Sets the system to validation mode
        self.eval()
        # Convert data to pytorch tensors and send to the computation device
        x, y = x.float().to(device=self.device), y.long().to(device=self.device)
        # Forward the data in the model  
        out = self.model.forward(x)

        # Compute loss
        loss = F.cosine_similarity(input=out, target=y)

        # Get argmax of predictions
        _, predicted = torch.max(out.data, 1)
        # Compute accuracy
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))
        return loss.cpu().data.numpy(), accuracy

    def save_model(self, model_save_dir, model_save_name, model_idx, best_validation_model_idx,
                   best_validation_model_acc):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        # Save network parameter and other variables.
        self.state['network'] = self.state_dict()
        # Save current best val idx
        self.state['best_val_model_idx'] = best_validation_model_idx
        # Save current best val acc
        self.state['best_val_model_acc'] = best_validation_model_acc
        # Save state at prespecified filepath
        torch.save(self.state, f=os.path.join(model_save_dir, 
                                    "{}_{}".format(model_save_name, 
                                    str(model_idx))))

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc 
        to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state, state['best_val_model_idx'], state['best_val_model_acc']

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        self.experiment.log_parameters(self.model.__dict__)
        wandb.watch(self, log="all")
        total_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

        # Start experiment
        with self.experiment.train():
            for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
                epoch_start_time = time.time()
                current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
                self.current_epoch = epoch_idx

                # Training
                with tqdm.tqdm(total=len(self.train_data)) as pbar_train:
                    # For each data batch
                    for idx, (x, y) in enumerate(self.train_data):
                        loss, accuracy = self.run_train_iter(x=x, y=y)
                        current_epoch_losses["train_loss"].append(loss)
                        current_epoch_losses["train_acc"].append(accuracy)
                        pbar_train.update(1)
                        pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

                # Validation
                with tqdm.tqdm(total=len(self.val_data)) as pbar_val:
                    # For each validation batch
                    for x, y in self.val_data:
                        loss, accuracy = self.run_validation_iter(x=x, y=y)
                        current_epoch_losses["val_loss"].append(loss)
                        current_epoch_losses["val_acc"].append(accuracy)
                        pbar_val.update(1)
                        pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
                
                val_mean_accuracy = np.mean(current_epoch_losses['val_acc'])

                # For early stopping
                if val_mean_accuracy > self.best_val_model_acc:
                    self.best_val_model_acc = val_mean_accuracy
                    self.best_val_model_idx = epoch_idx

                # Get mean of all metrics of current epoch metrics dict, 
                # to get them ready for storage and output on the terminal.
                for key, value in current_epoch_losses.items():
                    total_losses[key].append(np.mean(value))
                    self.experiment.log_metric(key, np.mean(value), step=epoch_idx)

                # Save statistics to stats file
                save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                                stats_dict=total_losses, current_epoch=i,
                                continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False)

                out_string = "_".join(["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
                
                # Report epoch metrics
                epoch_elapsed_time = time.time() - epoch_start_time
                epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
                print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")

                self.state['model_epoch'] = epoch_idx

                def quick_save(idx):
                    self.save_model(model_save_dir=self.experiment_saved_models,
                                    model_save_name="train_model", model_idx=idx,
                                    best_validation_model_idx=self.best_val_model_idx,
                                    best_validation_model_acc=self.best_val_model_acc)

                quick_save(epoch_idx)
                quick_save('latest')

        # Testing
        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, 
                        model_idx=self.best_val_model_idx,
                        model_save_name="train_model")

        current_test_losses = {"test_acc": [], "test_loss": []}
        with self.experiment.test():
            with tqdm.tqdm(total=len(self.test_data)) as pbar_test:
                for x, y in self.test_data:
                    # Compute loss and accuracy by running an evaluation step
                    loss, accuracy = self.run_validation_iter(x=x, y=y)
                    current_test_losses["test_loss"].append(loss)
                    current_test_losses["test_acc"].append(accuracy)
                    pbar_test.update(1)
                    pbar_test.set_description( "loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

            test_losses = {key: [np.mean(value)] for key, value in current_test_losses.items()}
            for key, value in current_test_losses.items():
                self.experiment.log_metric(key, value)

            save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                            stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        return total_losses, test_losses
