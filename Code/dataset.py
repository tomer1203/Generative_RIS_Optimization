import torch as T
from sklearn.decomposition import PCA
import numpy as np

import scipy.io
## DEBUG!!
from ChannelMatrixEvaluation import test_configurations_capacity
import torch
## DEBUG!!
from utils import get_simulation_grads,copy_with_gradients
from rate_model import capacity_loss
class abstract_dataset(T.utils.data.Dataset):
    """
    Abstract class for the dataset.
    This class is used to create a dataset for the generative model.
    """
    def __init__(self, batch_size, configuration_size, SoW_size, device, max_dataset_size,*args, **kwargs):
        self.batch_size = batch_size
        self.device = device
        self.configuration_size = configuration_size
        self.configurations = torch.zeros((0, configuration_size), device=device)
        self.sow_size = SoW_size
        self.sow = torch.zeros((0, SoW_size), device=device)
        self.max_dataset_size = max_dataset_size
        self.data_size = 0
        self.write_idx = 0
    def __len__(self):
        return self.data_size
    def generate_dataset(self, device, *args, **kwargs):
        for i in range(0,self.max_dataset_size,self.batch_size):
            self.add_new_batch(*self.generate_batch(self.batch_size, device, *args, **kwargs))
        self.data_size = len(self.configurations)

    def generate_sow(self, batch_size, device, *args, **kwargs):
        """
        Generate a batch of SoW.
        :param batch_size: The size of the batch.
        :param device: The device to use.
        :return: A batch of SoW.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    def generate_configuration(self, batch_size, device,*args, **kwargs):
        """
        Generate a batch of configurations.
        :param batch_size: The size of the batch.
        :param device: The device to use.
        :return: A batch of configurations.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def generate_batch(self, batch_size, device, save=False,*args, **kwargs):
        """
        Generate a batch of data.
        :param batch_size: The size of the batch.
        :param device: The device to use.
        :return: A tuple of configurations and SoW.
        """
        configurations = self.generate_configuration(batch_size, device,*args, **kwargs)
        sow = self.generate_sow(batch_size, device,*args, **kwargs)
        if save:
            self.add_new_batch(configurations, sow, batch_size)
        return (configurations, sow)

    def add_new_batch(self,configuration,sow,batch_size):
        """
        Add new items to the dataset.
        :param batch_size: The size of the batch.
        :param device: The device to use.
        :return: A batch of data.
        """
        if self.write_idx + batch_size > self.max_dataset_size:
            raise ValueError("using batch size which does not divide the dataset size evenly")
        # write the new batch to the dataset
        self.configurations[self.data_size:self.data_size + batch_size, :] = configuration
        self.sow[self.data_size:self.data_size + batch_size, :] = sow

        # update indexes
        self.write_idx += batch_size
        if self.write_idx == self.max_dataset_size: # check if we need to reset the write index
            self.write_idx = 0
        if self.data_size + batch_size <= self.max_dataset_size: # check if data reached the max size
            self.data_size += batch_size
        return 1

    def save(self,file_path, *args, **kwargs):
        """
        Save the dataset to a file.
        :param file_path: The path to the file.
        :return: None
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def load(self,file_sufix, *args, **kwargs):
        """
        Load the dataset from a file.
        :param file_sufix: The string suffix of the file name.
        :return: None
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class RISDataset(abstract_dataset):
    """
    Dataset for the RIS optimization problem.
    This dataset is used to train the generative model for the RIS optimization problem.
    """
    def __init__(self, batch_size, configuration_size, SoW_size, device, max_dataset_size ,virtual_batch_size=256):
        super().__init__(batch_size,configuration_size, SoW_size, device, max_dataset_size)
        self.device = device
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size


    def generate_configuration(self, batch_size, device=None):
        """
        Generate a batch of configurations.
        :param batch_size: The size of the batch.
        :param device: The device to use.
        :return: A batch of configurations.
        """
        if device is None:
            device = self.device
        return torch.rand([batch_size, self.configuration_size], device=device,dtype=torch.float64)

    def generate_sow(self, batch_size, device, modified_sow=True):
        """
        Generate a batch of SoW.
        :param batch_size: The size of the batch.
        :param device: The device to use.
        :return: A batch of SoW.
        """
        x_tx_orig = torch.tensor([0, 0, 0]).repeat(batch_size, 1).to(device).type(torch.float64)
        y_tx_orig = torch.tensor([4, 4.5, 5]).repeat(batch_size, 1).to(device).type(torch.float64)
        if not modified_sow:
            sow = torch.hstack([x_tx_orig, y_tx_orig])
            return sow
        tx_x_diff = 9.5 * torch.rand([batch_size, 3], device=device, dtype=torch.float64) - 3.3  # 19.5 *
        tx_y_diff = 11.5 * torch.rand([batch_size, 3], device=device, dtype=torch.float64) - 2.8
        tx_x, tx_y = x_tx_orig + tx_x_diff, y_tx_orig + tx_y_diff
        sow = torch.hstack([tx_x, tx_y])
        return sow


    def add_new_items(self,X,X_gradients,Y,Y_capacity):
        if X is None:
            return
        if torch.any(~torch.isfinite(X)):
            print("non finite value detected")
        self.configurations = T.vstack([self.x_data,X])
        self.gradients = T.vstack([self.gradients,X_gradients])
        self.y_data = T.vstack([self.y_data,Y])
        self.y_capacity = T.hstack([self.y_capacity,Y_capacity])


    def load(self,file_sufix, *args, **kwargs):
        """
        Load the dataset from a file.
        :param file_sufix: The string suffix of the file name.
        :return: None
        """
        RIS_file = "../Data/conditional_RISConfiguration"+file_sufix+".mat"  # "../Data/full_range_RISConfiguration.mat" # full_range_RISConfiguration
        tx_file = "../Data/conditional_transmitter_location"+file_sufix+".mat"

        # load the ris configuration from the file
        enclosure = {}
        scipy.io.loadmat(RIS_file, enclosure)
        ris_configs_np = enclosure["RISConfiguration"]

        # load the transmitter location from the file
        scipy.io.loadmat(tx_file, enclosure)
        tx_x_location = enclosure["x_tx_modified"]
        tx_y_location = enclosure["y_tx_modified"]
        # scipy.io.loadmat(tx_location_file + "conditional_transmitter_y_location.mat", enclosure)

        self.configurations = T.tensor(ris_configs_np, dtype=T.float32).to(self.device)
        self.tx_x = T.tensor(tx_x_location, dtype=T.float32, device=self.device)
        self.tx_y = T.tensor(tx_y_location, dtype=T.float32, device=self.device)
        self.sow = T.hstack([self.tx_x, self.tx_y])
        ldr = T.utils.data.DataLoader(self, batch_size=1, shuffle=True)

        return ldr

    def __len__(self):
        return len(self.configurations)//self.batch_size

    def __getitem__(self, idx):
        configurations = self.configurations[self.batch_size*idx:self.batch_size*(idx+1), :]  # or just [idx]
        sow = self.sow[int(idx//(self.virtual_batch_size/self.batch_size)),:]#(idx*batch_size)/256
        return (configurations,sow)  # tuple of two matrices
