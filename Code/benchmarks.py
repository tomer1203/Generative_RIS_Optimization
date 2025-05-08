import torch
from rate_model import capacity_loss
import numpy as np
from PhysFadPy import simulator_class

class benchmark_interface:
    """
    Interface for benchmark functions with basic functionality.
    """

    def __init__(self, name: str, simulation: simulator_class, device: torch.device):
        self.name = name
        self.simulation = simulation
        self.device = device

    def __call__(self, input_configuration, sow, snr_noise, *args, **kwargs) -> (torch.tensor, torch.tensor):
        """
        Evaluate the benchmark function with the given input configuration.
        returns the benchmark output and its evaluated performance.
        """
        channel = self.simulation(input_configuration, sow, *args, **kwargs)
        performance = capacity_loss(channel, P=None, sigmaN=snr_noise, list_out=False,device=self.device)
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

    def run(self, input_configuration,sow, learning_rate: float, num_of_iterations: int, snr_noise, *args, **kwargs):
        """
        Run the benchmark function for a specified number of iterations.
        """
        configuration_list = []
        performance_list = []
        current_configuration = input_configuration
        for i in range(num_of_iterations):
            # Evaluate the benchmark function
            output_channel, performance = self(current_configuration,sow, snr_noise, *args, **kwargs)
            # Evaluate the loss and gradient
            loss,gradient = self.loss_and_gradient(current_configuration, sow, output_channel, snr_noise, *args, **kwargs)
            # Perform a step in the optimization process
            current_configuration = self.step(current_configuration, gradient, learning_rate, *args, **kwargs)
            # Store the configuration and performance
            configuration_list.append(current_configuration)
            performance_list.append(performance)
            print(f"{self.name}: Iteration {i}: Performance = {performance}")
        return configuration_list, performance_list

class zo_benchmark(benchmark_interface):
    """
    Benchmark function using zeroth-order optimization.
    """

    def __init__(self, name: str, simulation: simulator_class, device: torch.device):
        super().__init__(name, simulation, device)
        self.name = name
        self.simulation = simulation
        self.device = device

    def loss_and_gradient(self, input_configuration,sow, channel=None, snr_noise=None, m=4, epsilon=0.1, broadcast_tx = True,*args,**kwargs) -> torch.tensor:
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
        batch_of_rand_vecs = self.generate_m_random_points_on_Nsphere(batch_size, m, N, self.device)
        f_x_plus_eps = torch.zeros((batch_size, m), device=self.device, dtype=torch.float64)
        f_x_minus_eps = torch.zeros((batch_size, m), device=self.device, dtype=torch.float64)

        with torch.no_grad():
            # TODO: broadcast this(currently broadcasting only the random_points)..
            #  I need to combine both the batches and the locations into the same dimension
            for i, rand_vectors in enumerate(batch_of_rand_vecs):
                if broadcast_tx:
                    current_sow = sow

                else:
                    current_sow = sow[i].unsqueeze(0)
                current_configuration = input_configuration[i].type(torch.float64)
                # get sample of points
                normalized_x_plus_epsilon = current_configuration + epsilon * rand_vectors
                normalized_x_minus_epsilon = current_configuration - epsilon * rand_vectors
                # test function on sample
                f_x_plus_eps[i] = self.simulation(normalized_x_plus_epsilon, current_sow, snr_noise=snr_noise, list_out=False, *args, **kwargs)[0]
                f_x_minus_eps[i] = self.simulation(normalized_x_minus_epsilon, current_sow, snr_noise=snr_noise, list_out=False, *args, **kwargs)[0]
                # TODO: and then I need to reseperate the batches and points into their own dimensions..
        return None,torch.sum((f_x_plus_eps - f_x_minus_eps).unsqueeze(2) * batch_of_rand_vecs / (2 * epsilon), dim=1) / m

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

# TODO: add the other benchmarks(random attempts, simulation based, etc.)
