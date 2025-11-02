from abc import ABC, abstractmethod
from dataset import physfad_dataset, sionna_dataset
from PhysFadPy import physfad_c,sionna_c
class abstract_factory(ABC):
    """
    Abstract class for the factory.
    This class is used to create a factory for the simulation and dataset
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None
        self.simulation = None
        self.device = None
    def create_dataset(self,config, batch_size, device, *args, **kwargs):
        """
        Creates a dataset. should be overridden by subclasses.
        :return: A dataset.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    def create_simulation(self,config, *args, **kwargs):
        """
        Create a simulation.
        :param args: The arguments to pass to the simulation.
        :param kwargs: The keyword arguments to pass to the simulation.
        :return: A simulation.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
class sionna_factory(abstract_factory):
    """
    Factory class for the physical fading model.
    This class is used to create a factory for the physical fading model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None
        self.simulation = None
        self.device = None
    def create_dataset(self,config, batch_size, device, *args, **kwargs):
        """
        Creates a dataset. should be overridden by subclasses.
        :param batch_size: The size of the batch.
        :param configuration_size: The size of the configuration.
        :param SoW_size: The size of the SoW.
        :param device: The device to use.
        :param max_dataset_size: The maximum size of the dataset.
        :param args: The arguments to pass to the dataset.
        :param kwargs: The keyword arguments to pass to the dataset.
        :return: A dataset.
        """
        # virtual_batch_size = kwargs["virtual_batch_size"] if "virtual_batch_size" in kwargs else 256
        return sionna_dataset(config, batch_size, config.input_size, config.sow_size, device, config.active_learning_mem_size, config.virtual_batch_size)
    def create_simulation(self, *args, **kwargs):
        """
        Create a simulation.
        :param args: The arguments to pass to the simulation.
        :param kwargs: The keyword arguments to pass to the simulation.
        :return: A simulation.
        """

        return sionna_c(*args, **kwargs)
class physfad_factory(abstract_factory):
    """
    Factory class for the physical fading model.
    This class is used to create a factory for the physical fading model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None
        self.simulation = None
        self.device = None
    def create_dataset(self,config, batch_size, device, *args, **kwargs):
        """
        Creates a dataset. should be overridden by subclasses.
        :param batch_size: The size of the batch.
        :param configuration_size: The size of the configuration.
        :param SoW_size: The size of the SoW.
        :param device: The device to use.
        :param max_dataset_size: The maximum size of the dataset.
        :param args: The arguments to pass to the dataset.
        :param kwargs: The keyword arguments to pass to the dataset.
        :return: A dataset.
        """
        # virtual_batch_size = kwargs["virtual_batch_size"] if "virtual_batch_size" in kwargs else 256
        return physfad_dataset(batch_size, config.input_size, config.sow_size, device, config.active_learning_mem_size, config.virtual_batch_size)
    def create_simulation(self, *args, **kwargs):
        """
        Create a simulation.
        :param args: The arguments to pass to the simulation.
        :param kwargs: The keyword arguments to pass to the simulation.
        :return: A simulation.
        """

        return physfad_c(*args, **kwargs)


