import torch
from rate_model import capacity_loss
import numpy as np
from PhysFadPy import simulator_class
from utils import copy_with_gradients

class benchmark_interface:
    """
    Interface for benchmark functions with basic functionality.
    """

    def __init__(self, name: str, simulation: simulator_class, device: torch.device, learning_rate: float = 0.01):
        self.name = name
        self.simulation = simulation
        self.device = device
        self.learning_rate = learning_rate
        self.require_grad = False
    def setup(self,input_configuration, *args, **kwargs):
        """
        Setup the benchmark with any necessary parameters.
        This can be used to initialize the benchmark with specific configurations.
        """
        return input_configuration
    def __call__(self, input_configuration, sow, snr_noise, *args, **kwargs) -> (torch.tensor, torch.tensor):
        """
        Evaluate the benchmark function with the given input configuration.
        returns the benchmark output and its evaluated performance.
        """
        performance,channel = self.simulation(input_configuration, sow, snr_noise=snr_noise,list_out=False, *args, **kwargs)
        return channel, performance
    def loss_and_gradient(self, input_configuration, sow, channel=None, snr_noise=None, *args, **kwargs) -> torch.tensor:
        """
        Evaluate the loss and gradient of the benchmark function with the given input configuration.
        returns the loss and its gradient.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def step(self, input_configuration, gradient, step_size: float, *args, **kwargs):
        """
        Perform a step in the optimization process. returns the updated configuration.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def run(self, input_configuration,sow, num_of_iterations: int, snr_noise, *args, **kwargs):
        """
        Run the benchmark function for a specified number of iterations.
        """
        configuration_list = []
        performance_list = []
        current_configuration = input_configuration
        if self.require_grad:
            current_configuration = copy_with_gradients(current_configuration,self.device)
        print(self.name, ": Starting benchmark run for", num_of_iterations, "iterations.")
        current_configuration = self.setup(current_configuration,*args, **kwargs)  # Setup the benchmark if needed
        for i in range(num_of_iterations):
            print(self.simulation.current_state)
            # Evaluate the benchmark function
            output_channel, performance = self(current_configuration,sow, snr_noise, *args, **kwargs)
            # Evaluate the loss and gradient
            loss,gradient = self.loss_and_gradient(current_configuration, sow, output_channel, snr_noise, *args, **kwargs)
            # Perform a step in the optimization process
            current_configuration = self.step(current_configuration, gradient, self.learning_rate, *args, **kwargs)
            # Store the configuration and performance
            configuration_list.append(current_configuration)
            performance_list.append(performance.detach().cpu().numpy())
            print(f"{self.name}: Iteration {i}: Performance = {performance}")
        return configuration_list, performance_list


class zo_benchmark(benchmark_interface):
    """
    Benchmark function using zeroth-order optimization.
    """

    def __init__(self, name : str, simulation: simulator_class, device: torch.device, learning_rate: float = 0.01, m=4, epsilon=0.1,broadcast_sow=True):
        super().__init__(name, simulation, device, learning_rate)
        self.name = name
        self.simulation = simulation
        self.device = device
        self.m = m
        self.epsilon = epsilon
        self.broadcast_sow = broadcast_sow

    def loss_and_gradient(self, input_configuration,sow, channel=None, snr_noise=None,*args,**kwargs) -> torch.tensor:
        """
        Evaluate the loss and gradient of the benchmark function with the given input configuration.
        returns the loss and its gradient.
        :param epsilon: The perturbation size.
        :param m: The number of random points to generate.
        :param broadcast_tx: Whether to broadcast the transmitter coordinates.
        returns only the gradient in this case.
        """
        # Implement the logic to compute the loss and gradient for zeroth-order optimization
        N = input_configuration.shape[-1]
        batch_size = input_configuration.shape[0]
        # m = kwargs["m"]
        # epsilon = kwargs["epsilon"]
        batch_of_rand_vecs = self.generate_m_random_points_on_Nsphere(batch_size, self.m, N, self.device)
        f_x_plus_eps = torch.zeros((batch_size, self.m), device=self.device, dtype=torch.float64)
        f_x_minus_eps = torch.zeros((batch_size, self.m), device=self.device, dtype=torch.float64)

        with torch.no_grad():
            # TODO: broadcast this(currently broadcasting only the random_points)..
            #  I need to combine both the batches and the locations into the same dimension
            for i, rand_vectors in enumerate(batch_of_rand_vecs):
                print("zo ",i," out of ", len(batch_of_rand_vecs))
                if self.broadcast_sow:
                    current_sow = sow

                else:
                    current_sow = sow[i].unsqueeze(0)
                current_configuration = input_configuration[i].type(torch.float64)
                # get sample of points
                normalized_x_plus_epsilon = current_configuration + self.epsilon * rand_vectors
                normalized_x_minus_epsilon = current_configuration - self.epsilon * rand_vectors
                # test function on sample
                f_x_plus_eps[i] = -self.simulation(normalized_x_plus_epsilon, current_sow, snr_noise=snr_noise, list_out=True, *args, **kwargs)[0]
                f_x_minus_eps[i] = -self.simulation(normalized_x_minus_epsilon, current_sow, snr_noise=snr_noise, list_out=True, *args, **kwargs)[0]
                # TODO: and then I need to reseperate the batches and points into their own dimensions..
        return None,torch.sum((f_x_plus_eps - f_x_minus_eps).unsqueeze(2) * batch_of_rand_vecs / (2 * self.epsilon), dim=1) / self.m

    def step(self, input_configuration, gradient, step_size: float, *args, **kwargs):
        """
        Perform a step in the optimization process. returns the updated configuration.
        """
        improved_configuration = input_configuration - step_size * gradient
        new_configuration = torch.clip(improved_configuration, 0, 1)
        return new_configuration
    def generate_m_random_points_on_Nsphere(self, batch_size, m, N, device):
        random_mat = np.random.random((batch_size, m, N)) * 2 - 1
        norm_mat = np.expand_dims(np.linalg.norm(random_mat, axis=2), axis=2)
        tensor_output = torch.tensor(random_mat / norm_mat, device=device)
        return tensor_output



class random_benchmark(benchmark_interface):
    """
    Benchmark function using random attempts.
    """

    def __init__(self, name: str, simulation: simulator_class, device: torch.device, learning_rate: float = 0.01):
        super().__init__(name, simulation, device, learning_rate)
        self.name = name
        self.simulation = simulation
        self.device = device
        self.best_configuration = None
        self.best_performance = None
    def __call__(self, input_configuration, sow, snr_noise, *args, **kwargs) -> (torch.tensor, torch.tensor):
        """
        Evaluate the benchmark function with the given input configuration.
        returns the benchmark output and its evaluated performance.
        """
        # Implement the logic for random attempts
        performance, channel = self.simulation(input_configuration, sow, snr_noise=snr_noise, list_out=False, *args, **kwargs)
        if self.best_configuration is None or performance > self.best_performance:
            self.best_configuration = input_configuration
            self.best_performance = performance

        return channel, self.best_performance
    def loss_and_gradient(self, input_configuration,sow, channel=None, snr_noise=None, *args, **kwargs) -> torch.tensor:
        """
        Evaluate the loss and gradient of the benchmark function with the given input configuration.
        returns the loss and its gradient.
        """
        return None, None  # not needed in this case, but required by the interface

    def step(self, input_configuration, gradient, step_size: float, *args, **kwargs):
        """
        Perform a step in the optimization process. returns the updated configuration.
        """
        # Implement the logic for a random step
        new_configuration = torch.rand_like(input_configuration)
        return new_configuration


class simulation_benchmark(benchmark_interface):
    """
    Benchmark function using a simulation-based approach.
    """

    def __init__(self, name: str, simulation: simulator_class, device: torch.device,learning_rate: float = 0.01):
        super().__init__(name, simulation, device, learning_rate)
        self.name = name
        self.simulation = simulation
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = None
        self.configurations = []  # Store configurations for optimization (and used later for simulation+noise benchmark)
        self.require_grad = True  # Indicate that this benchmark requires gradients in the tensors for optimization
    def setup(self,current_configuration, *args, **kwargs):
        """
        Setup the benchmark with any necessary parameters.
        This can be used to initialize the benchmark with specific configurations.
        """
        logit_configuration = copy_with_gradients(torch.special.logit(current_configuration),self.device)
        self.past_state = self.simulation.current_state
        self.simulation.change_environment("clean")
        return logit_configuration
    def __call__(self, input_configuration, sow, snr_noise, *args, **kwargs) -> (torch.tensor, torch.tensor):
        """
        Evaluate the benchmark function with the given input configuration.
        returns the benchmark output and its evaluated performance.
        """
        # Implement the logic for simulation-based optimization
        if self.optimizer is None:
            print("learning rate new ", self.learning_rate)
            self.optimizer = torch.optim.Adam([input_configuration], lr= self.learning_rate)
        inp_conf_norm = torch.nn.functional.sigmoid(input_configuration)
        performance, channel = self.simulation(inp_conf_norm, sow, snr_noise=snr_noise, *args, **kwargs)
        return channel, performance

    def loss_and_gradient(self, input_configuration,sow, channel=None, snr_noise=None, *args, **kwargs) -> torch.tensor:
        """
        Evaluate the loss and gradient of the benchmark function with the given input configuration.
        returns the loss and its gradient.
        """
        # Implement the logic for simulation-based optimization
        loss = -capacity_loss(channel, sigmaN=snr_noise, *args, **kwargs)
        loss.backward()
        return loss, None  # Gradient is computed internally in this case, but required by the interface

    def step(self, input_configuration, gradient, step_size: float, *args, **kwargs):
        """
        Perform a step in the optimization process. returns the updated configuration.
        """
        self.configurations.append(torch.nn.functional.sigmoid(input_configuration).clone().detach())
        self.optimizer.step()
        self.optimizer.zero_grad()
        return input_configuration  # The optimizer updates the configuration internally
    def run(self, input_configuration,sow, num_of_iterations: int, snr_noise, *args, **kwargs):
        """
        Run the benchmark function for a specified number of iterations.
        """
        return_values = super().run(input_configuration,sow, num_of_iterations, snr_noise, *args, **kwargs)  # Call the base class run method
        self.simulation.change_environment(self.past_state)  # Restore the previous state of the simulation
        return return_values
# TODO: add the other benchmarks(simulation with noise)
class simulation_noise_benchmark(benchmark_interface):
    """
    Benchmark function using a simulation-based approach with noise.
    """

    def __init__(self, name: str, simulation: simulator_class, device: torch.device, learning_rate: float = 0.01,simulation_benchmark_without_noise: benchmark_interface = None):
        super().__init__(name, simulation, device, learning_rate)
        self.name = name
        self.simulation = simulation
        self.device = device
        self.learning_rate = learning_rate
        self.simulation_benchmark_without_noise = simulation_benchmark_without_noise  # Stored configurations would be in here
        self.idx = 0  # Index for the configurations
    def setup(self,current_confiugration, *args, **kwargs):
        """
        Setup the benchmark with any necessary parameters.
        This can be used to initialize the benchmark with specific configurations.
        """
        self.past_state = self.simulation.current_state
        self.simulation.change_environment("noisy")
        return current_confiugration


    def __call__(self, input_configuration, sow, snr_noise, *args, **kwargs) -> (torch.tensor, torch.tensor):
        """
        Evaluate the benchmark function with the given input configuration.
        returns the benchmark output and its evaluated performance.
        """
        performance, channel = self.simulation(input_configuration, sow, snr_noise=snr_noise, list_out=False, *args, **kwargs)
        return channel, performance

    def loss_and_gradient(self, input_configuration,sow, channel=None, snr_noise=None, *args, **kwargs) -> torch.tensor:
        """
        Evaluate the loss and gradient of the benchmark function with the given input configuration.
        """
        return None, None  # No Gradients since using the same configurations from the noiseless channel.

    def step(self, input_configuration, gradient, step_size: float, *args, **kwargs):
        """
        Perform a step in the optimization process. returns the updated configuration.
        """
        self.idx += 1
        print("noisy idx", self.idx, len(self.simulation_benchmark_without_noise.configurations))

        if self.idx < len(self.simulation_benchmark_without_noise.configurations):
            new_configuration = self.simulation_benchmark_without_noise.configurations[self.idx].clone().detach()
        else:
            new_configuration = input_configuration
        return new_configuration # Use the configurations from the noiseless benchmark
    def run(self, input_configuration,sow, num_of_iterations: int, snr_noise, *args, **kwargs):
        """
        Run the benchmark function for a specified number of iterations.
        """
        return_values = super().run(input_configuration, sow, num_of_iterations, snr_noise, *args, **kwargs)
        self.simulation.change_environment(self.past_state)  # Restore the previous state of the simulation
        self.idx = 0
        return return_values
