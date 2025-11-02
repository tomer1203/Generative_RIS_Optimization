import torch
import scipy.special as scp
import numpy as np
import scipy.io
import math
import datetime
from memory_profiler import profile
import os
from matplotlib import pyplot as plt
from utils import LimitedSizeDict,to_tf,to_torch,reduce_sionna_shape
from copy import deepcopy
from contextlib import nullcontext
import concurrent.futures
import rate_model
import time
from functools import wraps
import sionna
import tensorflow as tf

from sionna import PI
from sionna.rt import load_scene, Transmitter, Receiver, RIS, PlanarArray, \
                      r_hat, normalize, Camera


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
# TODO: make the simulator generic(simulator_class) and make the physfad class inherit from it and fix all methods(*args, **kwargs)
class simulator_class:
    def __init__(self,config,device):
        self.config = config
        self.device = device
        self.current_state = "noisy"  # Default state of the environment

    def __call__(self,configuration,physical_SoW,*args, **kwargs):
        """
        This function is used to calculate the channel matrix H
        :param configuration: the configuration of the applied simulation
        :param physical_SoW: state of the world
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    def set_configuration(self,config):
        """
        This function is used to set the configuration of the simulation
        :return: None
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    def change_environment(self,state): # TODO: need to override this method in the subclasses
        """
        This function is used to change the environment
        :param state: the state of the environment can either be "clean" or "noisy"
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
class sionna_c(simulator_class):
    def __init__(self,config,device):
        super().__init__(config, device)
        self.name = "sionna"
        self.config = config
        self.device = device

    def __call__(self, configuration, physical_SoW, snr_noise=None,list_out=False, *args, **kwargs):
        torch_input = False
        H_ls = []
        cond_batch_size = physical_SoW.shape[0]
        conf_batch_size = configuration.shape[0]
        batch_size = conf_batch_size // cond_batch_size
        for i in range(len(physical_SoW)): # iterate over the given sow
            print(i, len(physical_SoW))
            self.scene.get("rx1").position = physical_SoW[i, 0:3]
            self.scene.get("rx2").position = physical_SoW[i, 3:6]
            ris_size = self.config.ris_num_modes*self.config.ris_num_rows*self.config.ris_num_cols
            if not (isinstance(configuration, tf.Tensor) or isinstance(configuration, tf.Variable)):
                torch_input = True
                configuration = to_tf(configuration) # note that this breaks the gradients for any torch input
            ris_shape = [self.config.ris_num_modes,self.config.ris_num_rows,self.config.ris_num_cols]
            for j in range(batch_size): # iterate each configuration of each sow
                ris1_amplitude = tf.reshape(configuration[i*batch_size+j, 0*ris_size:1*ris_size],ris_shape)
                ris1_phase     = tf.reshape(configuration[i*batch_size+j, 1*ris_size:2*ris_size],ris_shape)
                ris2_amplitude = tf.reshape(configuration[i*batch_size+j, 2*ris_size:3*ris_size],ris_shape)
                ris2_phase     = tf.reshape(configuration[i*batch_size+j, 3*ris_size:4*ris_size],ris_shape)
                self.ris1.amplitude_profile.values = ris1_amplitude / tf.sqrt(tf.reduce_mean(ris1_amplitude ** 2, axis=[1, 2], keepdims=True))
                self.ris1.phase_profile.values     = ris1_phase
                self.ris2.amplitude_profile.values = ris2_amplitude / tf.sqrt(tf.reduce_mean(ris2_amplitude ** 2, axis=[1, 2], keepdims=True))
                self.ris2.phase_profile.values     = ris2_phase

                paths = self.scene.compute_paths()
                a, tau = paths.cir()
                h_freq = sionna.channel.cir_to_ofdm_channel(sionna.channel.subcarrier_frequencies(self.config.output_size, self.config.subcarrier_spacing), a, tau, normalize=False)
                H_ls.append(reduce_sionna_shape(h_freq))
        H = tf.concat(H_ls,axis=0)
        rate = rate_model.capacity_loss_tf(H,sigmaN=snr_noise, list_out=list_out)
        if torch_input:
            rate = to_torch(rate)
        return rate,H
    def set_configuration(self, config):
        self.scene = load_scene(sionna.rt.scene.munich)
        self.scene.frequency = 3e9  # Carrier frequency [Hz]
        self.scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V")
        self.scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V")

        # Place a transmitter
        self.tx = Transmitter("tx", position=config.tx_position)
        self.scene.add(self.tx)

        # Place receivers
        self.rx1 = Receiver("rx1", position=config.rx1_position)
        self.scene.add(self.rx1)
        self.rx2 = Receiver("rx2", position=config.rx2_position)
        self.scene.add(self.rx2)

        # Place RIS
        self.ris1 = RIS(name="ris1",
                   position=config.ris1_location,
                   num_rows=config.ris_num_rows,
                   num_cols=config.ris_num_cols,
                   num_modes=config.ris_num_modes,
                   color=[0.8, 0, 0],
                   look_at=(self.tx.position + self.rx1.position) / 2)  # Look in between TX and RX1
        self.scene.add(self.ris1)

        self.ris2 = RIS(name="ris2",
                   position=config.ris2_location,
                   num_rows=config.ris_num_rows,
                   num_cols=config.ris_num_cols,
                   num_modes=config.ris_num_modes,
                   color=[0, 0, 0.8],
                   look_at=(self.tx.position + self.rx2.position) / 2)  # Look in between TX and RX2
        self.scene.add(self.ris2)
        # this camera is adjusted for the munich scene
        self.scene.add(Camera("cam",
                         position=[-0,250,150],
                         look_at=[60,30,-25]))
    def change_environment(self,state):
        """
        This function is used to change the environment
        :param state: the state of the environment can either be "clean" or "noisy"
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class physfad_c(simulator_class):
    def __init__(self,config,device):
        """
        This class is used to calculate the channel matrix H and the bessel matrix W
        :param config: the configuration of the simulation
        :param device: the device to use for the simulation
        """
        super().__init__(config, device)
        self.config = config
        self.device = device
        self.name = "physfad"
        self.W_dict = LimitedSizeDict(size_limit=128)
        self.clean_environment = False
        self.parameters = {}
        self.parameters_states = {}


    def __call__(self,ris_configuration_normalized,physical_SoW,recalculate_W=False,precalced_W=None,snr_noise=None, list_out=False,serial = False):
        """
        This function is used to calculate the channel matrix H and the bessel matrix W
        :param ris_configuration_normalized: the normalized configuration of the RIS
        :param physical_SoW: state of the world, the x and y coordinates of the transmitter
        :param recalculate_W: if True, the bessel matrix W will be recalculated
        :param precalced_W: if not None, the bessel matrix W will be used from the input
        returns the channel capacity and the channel matrix H
        """
        serial = True # TODO: remove this
        if serial or physical_SoW.shape[0] == 1:
            # print("serial call")
            return self.serial_call(ris_configuration_normalized,physical_SoW,recalculate_W=recalculate_W,precalced_W=precalced_W,snr_noise=snr_noise,list_out=list_out)
        cond_tx_x, cond_tx_y = physical_SoW[:, 0:3], physical_SoW[:, 3:6] # TODO: this should use some constants

        tx_size = cond_tx_x.shape[0]
        configuration_batch_size = ris_configuration_normalized.shape[0]
        batch_size = configuration_batch_size // tx_size
        with torch.no_grad() if not ris_configuration_normalized.requires_grad else nullcontext():
            if tx_size != 1:
                H = torch.zeros([configuration_batch_size, self.config.output_size, self.config.output_shape[0],
                                 self.config.output_shape[1]], dtype=torch.complex64)
                with concurrent.futures.ProcessPoolExecutor() as executer:
                    conf_ls = [confg.detach() for confg in [ris_configuration_normalized] * tx_size]
                    txx_ls = cond_tx_x.unsqueeze(1)
                    txy_ls = cond_tx_y.unsqueeze(1)
                    batch_ls = [batch_size] * tx_size
                    if precalced_W is None:
                        precalculate_W = [None] * tx_size
                    results = executer.map(self.batched_physfad, range(len(cond_tx_x)), conf_ls, txx_ls, txy_ls,
                                           batch_ls, precalculate_W)
                for i, (H_batch, W) in enumerate(results):
                    H[i * batch_size:(i + 1) * batch_size] = H_batch
                    # self.W_dict[(cond_tx_x[i].unsqueeze(0), cond_tx_y[i].unsqueeze(0))] = W

                return rate_model.capacity_loss(H, sigmaN=snr_noise, list_out=list_out, device=self.device), H
            else:
                H = self.get_H_and_W(ris_configuration_normalized, cond_tx_x, cond_tx_y)[0]
                return rate_model.capacity_loss(H, sigmaN=snr_noise, list_out=list_out, device=self.device), H
    def serial_call(self,ris_configuration_normalized,physical_SoW,recalculate_W=False,precalced_W=None,snr_noise=None, list_out=False):
        """
        This function is used to calculate the channel matrix H and the bessel matrix W
        :param ris_configuration_normalized: the normalized configuration of the RIS
        :param physical_SoW: state of the world, the x and y coordinates of the transmitter
        :param recalculate_W: if True, the bessel matrix W will be recalculated
        :param precalced_W: if not None, the bessel matrix W will be used from the input
        """
        cond_tx_x, cond_tx_y = physical_SoW[:,0:3], physical_SoW[:,3:6]# TODO: This should use some constants
        tx_size = cond_tx_x.shape[0]
        ris_configuration_size = ris_configuration_normalized.shape[0]
        batch_size = ris_configuration_size // tx_size
        with torch.no_grad() if not ris_configuration_normalized.requires_grad else nullcontext():
            if tx_size != 1:
                H = torch.zeros([ris_configuration_size, self.config.output_size, self.config.output_shape[0],
                                 self.config.output_shape[1]], dtype=torch.complex64)
                for i in range(len(cond_tx_x)):
                    batch_of_H = self.get_H_and_W(ris_configuration_normalized[i * batch_size:(i + 1) * batch_size], cond_tx_x[i].unsqueeze(0),
                                         cond_tx_y[i].unsqueeze(0),recalculate_W,precalced_W)[0]
                    if batch_size == 1:
                        batch_of_H = batch_of_H.unsqueeze(0)
                    H[i * batch_size:(i + 1) * batch_size] = batch_of_H

                return rate_model.capacity_loss(H, sigmaN=snr_noise, list_out=list_out, device=self.device), H
            H = self.get_H_and_W(ris_configuration_normalized, cond_tx_x, cond_tx_y,recalculate_W,precalced_W)[0]
        return rate_model.capacity_loss(H, sigmaN=snr_noise, list_out=list_out, device=self.device), H
        # cond_tx_x, cond_tx_y = physical_SoW[0], physical_SoW[1]
        # H = self.get_H_and_W(ris_configuration_normalized, cond_tx_x, cond_tx_y,recalculate_W=recalculate_W,precalced_W=precalced_W)[0]
        # return rate_model.capacity_loss(H, sigmaN=snr_noise, list_out=list_out, device=self.device), H
    def batched_physfad(self,i, ris_configuration, tx_x, tx_y, batch_size, precalculate_W=None):
        batch_of_H, W = self.get_H_and_W(ris_configuration[i * batch_size:(i + 1) * batch_size], tx_x, tx_y,
                                precalced_W=precalculate_W)
        # print(i)
        if batch_size == 1:
            batch_of_H = batch_of_H.unsqueeze(0)
        return batch_of_H.detach(), W
    def get_H_and_W(self,ris_configuration_normalized,cond_tx_x,cond_tx_y,recalculate_W=False,precalced_W=None):
        if not torch.is_tensor(ris_configuration_normalized):
            ris_configuration_normalized = torch.tensor(ris_configuration_normalized, device=self.device)

        ris_configuration_full = self.fill_ris_config(ris_configuration_normalized)
        # ris_configuration = ris_configuration_full
        ris_configuration = self.scale_output_to_range(ris_configuration_full)

        fres_ris_c  = ris_configuration[:, self.N_RIS*0:self.N_RIS*1]
        chi_ris_c   = ris_configuration[:, self.N_RIS*1:self.N_RIS*2]
        gamma_ris_c = ris_configuration[:, self.N_RIS*2:self.N_RIS*3]
        # change the parameters to the new configuration
        current_parameters = deepcopy(self.parameters)
        current_parameters["x_tx"] = cond_tx_x
        current_parameters["y_tx"] = cond_tx_y
        current_parameters["fres_ris"] = fres_ris_c
        current_parameters["chi_ris"] = chi_ris_c
        current_parameters["gamma_ris"] = gamma_ris_c
        # W is given as input (precalculated outside the class)
        if precalced_W is not None:
            W = precalced_W
        # W is already calculated in the dictionary (cached)
        elif any([not torch.any(x-cond_tx_x) for x in list(self.W_dict.keys())]) and not recalculate_W:
            # print("reusing W from cache")
            W = self.W_dict[cond_tx_x]
        # W is not calculated yet, so we need to calculate it and cache it
        else:
            # print("calculating W")
            W = self.get_bessel_w(current_parameters, self.device)
            self.W_dict[cond_tx_x] = W
        if self.config.test_woodbury_transform and current_parameters["first_calculation_in_environment"]:
            self.parameters["first_calculation_in_environment"] = False
            self.parameters["W_inv"],self.parameters["default_W"] = self.calculate_initial_W_inverse(current_parameters,W)
        H = self.GetH(current_parameters, W)

        return H,W
    def clear_bessel_mem(self):
        self.W_dict = LimitedSizeDict(size_limit=128)
        self.parameters["first_calculation_in_environment"] = True
    # def generate_tx_location(self,size,device,modify=True):
    #     x_tx_orig = torch.tensor([0, 0, 0]).repeat(size, 1).to(device).type(torch.float64)
    #     y_tx_orig = torch.tensor([4, 4.5, 5]).repeat(size, 1).to(device).type(torch.float64)
    #     if not modify:
    #         return x_tx_orig, y_tx_orig
    #     tx_x_diff = 19.5 * torch.rand([size, 3], device=device,dtype=torch.float64) - 3.3 # 19.5 *
    #     tx_y_diff = 11.5 * torch.rand([size, 3], device=device,dtype=torch.float64) - 2.8
    #     tx_x, tx_y = x_tx_orig + tx_x_diff, y_tx_orig + tx_y_diff
    #     return tx_x,tx_y
    def change_rx_location(self,x_rx_new,y_rx_new):
        self.parameters["x_rx"] = x_rx_new
        self.parameters["y_rx"] = y_rx_new
        self.clear_bessel_mem()

    def plot_environment(self,sow=None,y_scaler = 1,x_scaler=1,y_shift=0,x_shift=0,show=True):
        cond_tx_x, cond_tx_y = sow[:, 0:3], sow[:, 3:6] # TODO: this should use some constants

        plt.scatter(x_scaler*(self.parameters["x_env"]+x_shift),y_scaler*(self.parameters["y_env"]+y_shift))
        if sow is not None:
            plt.scatter(cond_tx_x,cond_tx_y)
        if show:
            plt.show()
    def change_environment(self,state):
        """
        This function is used to change the environment
        :param state: the state of the environment can either be "clean" or "noisy"
        """
        self.parameters_states[self.current_state] = deepcopy(self.parameters) # saves any changes made to the current environment
        self.parameters = self.parameters_states[state] # loads the parameters of the new environment
        if state != self.current_state:
            print(f"Changing environment from {self.current_state} to {state}")
            self.current_state = state
            self.parameters["first_calculation_in_environment"] = True
            self.clear_bessel_mem() # only delete the bessel memory if the environment was actually changed



    def set_configuration(self,config):
        self.parameters["freq"] = torch.tensor(np.linspace(0.9, 1.1, 120));

        ## Configurable Dipole Properties
        ## Transmitters ##
        # locations
        self.parameters["x_tx"] = torch.tensor([0, 0, 0]).unsqueeze(0).to(self.device).type(torch.float64)
        self.parameters["y_tx"] = torch.tensor([4, 4.5, 5]).unsqueeze(0).to(self.device).type(torch.float64)
        # dipole properties
        self.parameters["fres_tx"] = torch.tensor([1, 1, 1]).unsqueeze(0).to(self.device).type(torch.float64)
        self.parameters["chi_tx"] = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(0).to(self.device).type(torch.float64)
        self.parameters["gamma_tx"] = torch.tensor([0, 0, 0]).unsqueeze(0).to(self.device).type(torch.float64)

        ##  Receivers ##
        # locations
        self.parameters["x_rx"] = torch.tensor([15, 15, 15, 15]).unsqueeze(0).to(self.device).type(torch.float64)
        self.parameters["y_rx"] = torch.tensor([11, 11.5, 12, 12.5]).unsqueeze(0).to(self.device).type(torch.float64)
        # properties
        self.parameters["fres_rx"] = torch.tensor([1, 1, 1, 1]).unsqueeze(0).to(self.device).type(torch.float64)
        self.parameters["chi_rx"] = torch.tensor([0.5, 0.5, 0.5, 0.5]).unsqueeze(0).to(self.device).type(torch.float64)
        self.parameters["gamma_rx"] = torch.tensor([0, 0, 0, 0]).unsqueeze(0).to(self.device).type(torch.float64)

        enclosure = {}
        enclosure_clean = {}
        # If never generated a noisy environment then generate a new one
        if os.path.isfile("..//Data//"+self.config.environment_file_name+"Noised.mat"):
            scipy.io.loadmat("..//Data//"+self.config.environment_file_name+".mat", enclosure_clean)
        else:
            print("generating new noisy room")
            scipy.io.loadmat("..//Data//"+self.config.environment_file_name+".mat", enclosure_clean)
            self.x_env_clean = torch.tensor(enclosure_clean['x_env']).to(self.device).type(torch.float64)
            self.y_env_clean = torch.tensor(enclosure_clean['y_env']).to(self.device).type(torch.float64)
            total_env = (self.x_env_clean+self.y_env_clean)/2
            mean_env_power = torch.sqrt((total_env**2).mean())
            noise_power = mean_env_power*self.config.environment_noise_power
            x_env = self.x_env_clean + torch.normal(0,noise_power*torch.ones_like(self.x_env_clean))
            y_env = self.y_env_clean + torch.normal(0,noise_power*torch.ones_like(self.y_env_clean))
            plt.scatter(x_env,y_env)
            plt.show()
            scipy.io.savemat("..//Data//"+self.config.environment_file_name+"Noised.mat", {"x_env": x_env.cpu().detach().numpy(),
                                                                                           "y_env": y_env.cpu().detach().numpy()})

        scipy.io.loadmat("..//Data//"+self.config.environment_file_name+"Noised.mat", enclosure)
        self.parameters["x_env"] = torch.tensor(enclosure['x_env']).to(self.device).type(torch.float64) # noisy environment
        self.parameters["y_env"] = torch.tensor(enclosure['y_env']).to(self.device).type(torch.float64)
        self.parameters["x_env_clean"] = torch.tensor(enclosure_clean['x_env']).to(self.device).type(torch.float64)
        self.parameters["y_env_clean"] = torch.tensor(enclosure_clean['y_env']).to(self.device).type(torch.float64)
        self.parameters["fres_env"] = 10 * torch.ones(self.parameters["x_env"].shape).to(self.device).type(torch.float64)
        self.parameters["chi_env"] = 50 * torch.ones(self.parameters["x_env"].shape).to(self.device).type(torch.float64)
        self.parameters["gamma_env"] = 0 * torch.ones(self.parameters["x_env"].shape).to(self.device).type(torch.float64)
        self.parameters["first_calculation_in_environment"] = True
        self.parameters_states["noisy"] = deepcopy(self.parameters)
        RIS_loc = {}
        scipy.io.loadmat("..//PhysFad//ExampleRIS.mat", RIS_loc)
        self.parameters["x_ris"] = torch.tensor(RIS_loc['x_ris']).to(self.device).type(torch.float64)
        self.parameters["y_ris"] = torch.tensor(RIS_loc['y_ris']).to(self.device).type(torch.float64)
        ris_num_samples = 3
        N_RIS = len(self.parameters["x_ris"][0])
        self.N_RIS = N_RIS
        self.N_RIS_PARAMS = 3*N_RIS

        self.parameters["x_ris"] = self.parameters["x_ris"][0].unsqueeze(0).to(self.device).type(torch.float64)
        self.parameters["y_ris"] = self.parameters["y_ris"][0].unsqueeze(0).to(self.device).type(torch.float64)

        # save the parameters in the noisy room state(which is also the default state)
        self.parameters_states["noisy"] = deepcopy(self.parameters)
        # save the parameters in the clean room state
        clean_parameters = deepcopy(self.parameters_states["noisy"])
        clean_parameters["x_env"] = clean_parameters["x_env_clean"]
        clean_parameters["y_env"] = clean_parameters["y_env_clean"]
        self.parameters_states["clean"] = clean_parameters

        # RISConfiguration = np.loadtxt("RandomConfiguration.txt")
        # self.parameters["fres_ris"] = torch.tensor(RISConfiguration[0:88]).unsqueeze(0).to(self.device).type(torch.float64)
        # self.parameters["chi_ris"] = torch.tensor(RISConfiguration[88:176]).unsqueeze(0).to(self.device).type(torch.float64)
        # self.parameters["gamma_ris"] = torch.tensor(RISConfiguration[176:264]).unsqueeze(0).to(self.device).type(torch.float64)


    def besselh(self,order,kind=2,z=0,scale=0):
        # return besselj(0,z)-torch.tensor([1j])*bessely(0,z);
        return scp.hankel2(order, z)

    def get_bessel_w_rx_change(self,rx_location):
        x_rx, y_rx = rx_location
        updated_rx_parameters = deepcopy(self.parameters)
        updated_rx_parameters["x_rx"] = x_rx
        updated_rx_parameters["y_rx"] = y_rx
        self.clear_bessel_mem()
        W = self.get_bessel_w(updated_rx_parameters, self.device)
        return W

    def get_bessel_w(self,parameters,device):
        """
            This function is a reimplementation of the PhysFad algorithm to calculate the bessel matrix W.
        """
        k = 2 * torch.pi * parameters["freq"]
        x = torch.cat([parameters["x_tx"], parameters["x_rx"], parameters["x_ris"], parameters["x_env"]], 1)
        y = torch.cat([parameters["y_tx"], parameters["y_rx"], parameters["y_ris"], parameters["y_env"]], 1)

        N_T = len(parameters["x_tx"][0])
        N_R = len(parameters["x_rx"][0])
        N_E = len(parameters["x_env"][0])
        N_RIS = len(parameters["x_ris"][0])
        N = N_T + N_R + N_E + N_RIS
        H = torch.zeros([len(parameters["freq"]), N_R, N_T], dtype=torch.complex64, device=device)
        pi = torch.pi
        W = torch.zeros([len(parameters["freq"]),N,N],dtype=torch.complex64,device=device)
        for f in range(len(parameters["freq"])):
            x_diff = torch.zeros([N, N], dtype=torch.float64, device=device)
            y_diff = torch.zeros([N, N], dtype=torch.float64, device=device)
            for l in range(N):
                xl_vec = x[0, l] * torch.ones([1, N], dtype=torch.float64, device=device)
                yl_vec = y[0, l] * torch.ones([1, N], dtype=torch.float64, device=device)
                x_diff[l, :] = x - xl_vec
                y_diff[l, :] = y - yl_vec
            BesselInp = k[f] * torch.sqrt(x_diff ** 2 + y_diff ** 2)
            # BesselInp = torch.sqrt(x_diff**2+y_diff**2)
            BesselOut = torch.tensor(self.besselh(0, 2, BesselInp.cpu().detach().numpy()), device=device)
            W[f] = 1j * (k[f] ** 2 / 4) * BesselOut
        return W

    def GetH_batched(self, parameters, W_full):
        """
            This function is a reimplementation of the PhysFad algorithm to calculate the channel matrix H.
            It is optimized to handle batches of parameters for faster computation. This makes it less readable but more efficient for large datasets.
            for more details see the original PhysFad paper
        """
        print("batched Physfad")
        epsilon = 0.00000001
        k = 2 * torch.pi * parameters["freq"]
        batch_size = parameters["fres_ris"].shape[0]
        fres = torch.cat([parameters["fres_tx"].repeat(batch_size,1), parameters["fres_rx"].repeat(batch_size,1), parameters["fres_ris"], parameters["fres_env"].repeat(batch_size,1)], 1)
        chi = torch.cat([parameters["chi_tx"].repeat(batch_size,1), parameters["chi_rx"].repeat(batch_size,1), parameters["chi_ris"], parameters["chi_env"].repeat(batch_size,1)], 1)
        gamma = torch.cat([parameters["gamma_tx"].repeat(batch_size,1), parameters["gamma_rx"].repeat(batch_size,1), parameters["gamma_ris"], parameters["gamma_env"].repeat(batch_size,1)], 1)

        N_T = len(parameters["x_tx"][0])
        N_R = len(parameters["x_rx"][0])
        N_E = len(parameters["x_env"][0])
        N_RIS = len(parameters["x_ris"][0])
        N = N_T + N_R + N_E + N_RIS
        pi = torch.pi
        k2 = (torch.pow(k, 2)).repeat(batch_size,1).to(self.device)
        two_pi = 2 * pi
        two_pi_freq = (two_pi * parameters["freq"]).repeat(batch_size,1).to(self.device)
        two_pi_freq2 = torch.pow(two_pi_freq, 2)

        chi2 = torch.pow(chi, 2)+epsilon
        two_pi_fres2 = torch.pow((two_pi * fres), 2)

        inv_alpha = (two_pi_fres2.unsqueeze(2) - two_pi_freq2.unsqueeze(1)) / (chi2.unsqueeze(2)) + 1j * ((
                    k2.unsqueeze(1) / 4) + two_pi_freq.unsqueeze(1) * gamma.unsqueeze(2) / chi2.unsqueeze(2))
        inv_alpha = inv_alpha.type(torch.complex64)

        W = W_full.clone().repeat(batch_size,1,1,1)
        # width = W.size(0)
        Mask = torch.eye(W.size(2)).repeat(batch_size,len(parameters["freq"]), 1, 1).bool()
        W[Mask] = inv_alpha.permute(0,2,1).reshape(-1)
        W_diag_elem = torch.diagonal(W, dim1=-2, dim2=-1)
        W_diag_matrix = torch.zeros(W.shape, dtype=torch.complex64, device=self.device)
        W_diag_matrix.diagonal(dim1=-2, dim2=-1).copy_(W_diag_elem)
        if self.config.test_woodbury_transform:
            start = time.time()
            V = torch.linalg.solve_ex(W, W_diag_matrix)[0]
            end = time.time()
            print("original solve time:  ",end - start)
            start = time.time()
            dW = W - self.parameters["default_W"].repeat(batch_size,1,1,1)
            # dW[~Mask] = 0
            if dW[Mask].abs().max() < 1e-10:
                print("dW min value: ", dW[Mask].abs().min())
                dW[Mask] = dW[Mask] + epsilon
            W_inv = self.inverse_W_woodbury(self.parameters["W_inv"].repeat(batch_size,1,1,1), dW, N_T + N_R + N_RIS, N, 1)
            end = time.time()
            # print(end - start)
            V_tag = W_inv @ W_diag_matrix
            print("error between woodbury and original solve: ", torch.sum(torch.abs(H - V_tag[:,:, N_T: (N_T + N_R), 0: N_T])))
        else:
            V = torch.linalg.solve_ex(W, W_diag_matrix)[0]
        H = V[:,:, N_T: (N_T + N_R), 0: N_T]
        return H
    @timeit
    def inverse_W_woodbury(self,W_orig_inv,dW,m,N,batch_size):
        """
        calculate the inverse (W_orig+dW)^-1 using the woodbury formula
        inputs: for N>m
        - W = BxFxNxN or FxNxN matrix
        - W_inv = W^(-1) BxFxNxN or NxN matrix
        - dW = BxFxNxN or NxN diagonal matrix filled up to row m
        - m = the rank of dW
        - N = the Height/Width of the W matrix
        - B/batch_size = batch_size
        this whole inversion takes O(m^3) time compared to O(N^3) time
        returns (W+dW)^-1
        """
        if len(W_orig_inv.shape) == 4: # BxFxNxN matrix's (B=batch,F=freq,N=mat_size)
            F = W_orig_inv.shape[1]
            I = torch.eye(m,dtype=torch.complex64).repeat(batch_size,F,1,1)
            U,V = torch.zeros([batch_size,F,N,m],dtype=torch.complex64),torch.zeros([batch_size,F,m,N],dtype=torch.complex64)
            U[:,:,0:m,0:m] = I
            C = dW[:,:,0:m,0:m]
            V[:,:,0:m,0:m] = I
        elif len(W_orig_inv.shape) == 3: # FxNxN matrix's (F=freq,N=mat_size)
            F = W_orig_inv.shape[0]
            I = torch.eye(m,dtype=torch.complex64).repeat(F,1,1) # Fxmxm
            U, V = torch.zeros([F,N,m],dtype=torch.complex64), torch.zeros([F,m,N],dtype=torch.complex64)
            U[:,0:m, 0:m] = I
            C = dW[:,0:m,0:m]
            V[:,0:m, 0:m] = I
        else:
            assert False, "W_orig_inv has invalid shape"
        return self.woodbury_matrix_inversion(W_orig_inv,U,C,V)

    def woodbury_matrix_inversion(self,R_inv,U,C,V):
        """
           calculate the inverse: (R+UCV)^-1 using Sherman–Morrison–Woodbury formula
           (R+UCV)^-1 = R^(-1) - R^(-1)*U*(C^(-1) +VR^(-1)U)^(-1)*VR^(-1)
            inputs: for N>m
            - R = NxN matrix
            - R_inv = R^(-1) NxN matrix
            - U = mxN matrix
            - V = Nxm matrix
            - C = mxm matrix
            this whole inversion takes O(m^3) time compared to O(N^3) time
            returns (R+UCV)^-1
        """
        C_inv = torch.linalg.pinv(C)
        reduced_rank_inv = torch.linalg.solve_ex((C_inv + V @ R_inv @ U), V @ R_inv)[0] # this is an O(m^3) operation
        return R_inv - R_inv @ U @ reduced_rank_inv
    @timeit
    def solve1(self, W, W_diag_matrix):
        """
        This function is used to solve the equation W * V = W_diag_matrix
        :param W: the bessel matrix W
        :param W_diag_matrix: the diagonal matrix of W
        :return: the solution V
        """
        try:
            V = torch.linalg.solve_ex(W, W_diag_matrix)
            return V
        except RuntimeError as e:
            print("Error in solving the equation:", e)
            return None
    @timeit
    def solve2(self, W, W_diag_matrix):
        """
        This function is used to solve the equation W * V = W_diag_matrix
        :param W: the bessel matrix W
        :param W_diag_matrix: the diagonal matrix of W
        :return: the solution V
        """
        n = W.shape[0]
        try:
            LD1,pivots1 = torch.linalg.ldl_factor_ex(W[:n//2],hermitian=True)
            V1 = torch.linalg.ldl_solve(LD1, pivots1, W_diag_matrix[:n//2])
            LD2, pivots2 = torch.linalg.ldl_factor_ex(W[n // 2:], hermitian=True)
            V2 = torch.linalg.ldl_solve(LD2, pivots2, W_diag_matrix[n // 2:])
            V = torch.cat([V1, V2], dim=0)
            return V
        except RuntimeError as e:
            print("Error in solving the equation:", e)
            return None
    def calculate_initial_W_inverse(self,parameters, W_full):
        # print("normal Physfad")
        if parameters["fres_ris"].shape[0] != 1:
            parameters["fres_ris"] = parameters["fres_ris"][0].unsqueeze(0)
            parameters["chi_ris"] = parameters["chi_ris"][0].unsqueeze(0)
            parameters["gamma_ris"] = parameters["gamma_ris"][0].unsqueeze(0)

        epsilon = 0.00000001
        k=2*torch.pi*parameters["freq"]
        # x = torch.cat([x_tx, x_rx, x_env, x_ris],1)
        # y = torch.cat([y_tx, y_rx, y_env, y_ris],1)

        fres = torch.cat([parameters["fres_tx"], parameters["fres_rx"], parameters["fres_ris"], parameters["fres_env"]],1)
        chi = torch.cat([parameters["chi_tx"], parameters["chi_rx"], parameters["chi_ris"], parameters["chi_env"]],1)
        gamma = torch.cat([parameters["gamma_tx"], parameters["gamma_rx"], parameters["gamma_ris"], parameters["gamma_env"]],1)

        N_T   = len(parameters["x_tx"][0])
        N_R   = len(parameters["x_rx"][0])
        N_E   = len(parameters["x_env"][0])
        N_RIS = len(parameters["x_ris"][0])
        N = N_T + N_R + N_E + N_RIS
        pi = torch.pi
        k2 = (torch.pow(k, 2)).to(self.device)
        two_pi = 2 * pi
        two_pi_freq = (two_pi * parameters["freq"]).to(self.device)
        two_pi_freq2 = torch.pow(two_pi_freq, 2)
        chi2 = torch.pow(chi[0, :], 2)+epsilon
        gamma_ = gamma[0,:]
        two_pi_fres2 = torch.pow((two_pi * fres[0, :]), 2)

        inv_alpha = (two_pi_fres2.unsqueeze(1) - two_pi_freq2.unsqueeze(0)) / (chi2.unsqueeze(1)) + 1j * ((k2.unsqueeze(0) / 4) + two_pi_freq.unsqueeze(0) * gamma_.unsqueeze(1) / chi2.unsqueeze(1))
        inv_alpha = inv_alpha.type(torch.complex64)
        W = W_full.clone()
        # width = W.size(0)
        Mask = torch.eye(W.size(1)).repeat(len(parameters["freq"]), 1, 1).bool()
        W[Mask] = inv_alpha.T.reshape(-1)
        first_W_inv = torch.linalg.inv(W)
        return first_W_inv,W

    def GetH(self,parameters, W_full):

        if parameters["fres_ris"].shape[0]!=1:
            return self.GetH_batched(parameters, W_full)

        # print("normal Physfad")
        epsilon = 0.00000001
        k=2*torch.pi*parameters["freq"]
        # x = torch.cat([x_tx, x_rx, x_env, x_ris],1)
        # y = torch.cat([y_tx, y_rx, y_env, y_ris],1)
        fres = torch.cat([parameters["fres_tx"], parameters["fres_rx"],parameters["fres_ris"], parameters["fres_env"]],1)
        chi = torch.cat([parameters["chi_tx"], parameters["chi_rx"], parameters["chi_ris"], parameters["chi_env"]],1)
        gamma = torch.cat([parameters["gamma_tx"], parameters["gamma_rx"], parameters["gamma_ris"], parameters["gamma_env"]],1)

        N_T   = len(parameters["x_tx"][0])
        N_R   = len(parameters["x_rx"][0])
        N_E   = len(parameters["x_env"][0])
        N_RIS = len(parameters["x_ris"][0])
        N = N_T + N_R + N_E + N_RIS
        H = torch.zeros([len(parameters["freq"]),N_R,N_T],dtype=torch.complex64,device=self.device)
        pi = torch.pi
        k2 = (torch.pow(k, 2)).to(self.device)
        two_pi = 2 * pi
        two_pi_freq = (two_pi * parameters["freq"]).to(self.device)
        two_pi_freq2 = torch.pow(two_pi_freq, 2)
        chi2 = torch.pow(chi[0, :], 2)+epsilon
        gamma_ = gamma[0,:]
        two_pi_fres2 = torch.pow((two_pi * fres[0, :]), 2)

        inv_alpha = (two_pi_fres2.unsqueeze(1) - two_pi_freq2.unsqueeze(0)) / (chi2.unsqueeze(1)) + 1j * ((k2.unsqueeze(0) / 4) + two_pi_freq.unsqueeze(0) * gamma_.unsqueeze(1) / chi2.unsqueeze(1))
        inv_alpha = inv_alpha.type(torch.complex64)
        W = W_full.clone()
        # width = W.size(0)
        Mask = torch.eye(W.size(1)).repeat(len(parameters["freq"]), 1, 1).bool()
        W[Mask] = inv_alpha.T.reshape(-1)
        W_diag_elem = torch.diagonal(W,dim1=-2,dim2=-1)
        W_diag_matrix = torch.zeros(W.shape,dtype=torch.complex64,device=self.device)
        W_diag_matrix.diagonal(dim1=-2,dim2=-1).copy_(W_diag_elem)
        if self.config.test_woodbury_transform:
            start = time.time()
            V = torch.linalg.solve(W, W_diag_matrix)
            end = time.time()
            print(end - start)
            start = time.time()
            dW = W - self.parameters["default_W"]
            # dW[~Mask] = 0
            # print("dW max value: ", dW[Mask].abs().max())
            if dW[Mask].abs().max() < 1e-10:
                print("dW max value: ", dW[Mask].abs().max())
                dW[Mask] = dW[Mask]+epsilon/20
            W_inv = self.inverse_W_woodbury(self.parameters["W_inv"], dW, N_T + N_R + N_RIS, N, 1)
            end = time.time()
            print(end - start)
            V_tag = W_inv @ W_diag_matrix
            print("error between woodbury and original solve: ",
                  torch.sum(torch.abs(H - V_tag[:, N_T: (N_T + N_R), 0: N_T])))
        else:
            V = torch.linalg.solve(W, W_diag_matrix)
        H = V[:,N_T: (N_T + N_R), 0: N_T]
        # H = V_tag[:,N_T: (N_T + N_R), 0: N_T]

        return H

    def fill_ris_config(self,fres):
        '''
            in case the input is only resonant frequency we need to wrap it with all the rest of the ris configuration
        '''
        if len(fres.shape) == 1:  # single element,no batch
            batch_size = 1
            number_of_elements = fres.shape[0]
            fres = fres.unsqueeze(0)
            if len(fres) == self.N_RIS_PARAMS:  # filling not needed, return original
                return fres
        else:
            batch_size = fres.shape[0]
            number_of_elements = fres.shape[1]

            if fres.shape[1] == self.N_RIS_PARAMS:  # filling not needed, return original
                return fres
        # No gradient Required since we are only optimizing the Resonant Frequency
        chi_ris = 0.2 * torch.ones((batch_size, number_of_elements), dtype=torch.float64, device=self.device,requires_grad=fres.requires_grad)
        gamma_ris = 0 * torch.zeros((batch_size, number_of_elements), dtype=torch.float64, device=self.device,requires_grad=fres.requires_grad)

        return torch.hstack([fres, chi_ris, gamma_ris])

    def scale_output_to_range(self, normalized_output):
        batch_size = normalized_output.shape[0]
        num_config = normalized_output.shape[1]
        assert num_config % 3 == 0, "configuration count should be divisible by three"
        num_ris_elements = num_config // 3
        fres_output = normalized_output[:, 0 * num_ris_elements:1 * num_ris_elements]
        chi_output = normalized_output[:, 1 * num_ris_elements:2 * num_ris_elements]
        gamma_output = normalized_output[:, 2 * num_ris_elements:3 * num_ris_elements]
        fres_output_rescaled = fres_output * self.config.fres_max_range
        chi_output_rescaled = chi_output * self.config.chi_max_range
        gamma_output_rescaled = gamma_output * self.config.gamma_max_range
        return torch.hstack([fres_output_rescaled, chi_output_rescaled, gamma_output_rescaled])

