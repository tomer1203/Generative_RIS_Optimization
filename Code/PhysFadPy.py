import torch
import scipy.special as scp
import numpy as np
import scipy.io
import math
import datetime
from memory_profiler import profile
import os
from matplotlib import pyplot as plt
from utils import LimitedSizeDict
from copy import deepcopy
class physfad_c():
    def __init__(self,config,device):
        self.config = config
        self.device = device
        self.W_dict = LimitedSizeDict(size_limit=128)
        self.clean_environment = False
        self.parameters = {}


    def __call__(self,ris_configuration_normalized,cond_tx_x,cond_tx_y,recalculate_W=False,precalced_W=None):
        # (freq, x_tx, y_tx, parameters["fres_tx"], chi_tx, gamma_tx,
        #  x_rx, y_rx, fres_rx, chi_rx, gamma_rx,
        #  x_env, y_env, fres_env, chi_env, gamma_env, x_ris_c, y_ris_c) = self.parameters
        # if torch.any(~torch.isfinite(ris_configuration_normalized)):
        #     print("in physfad H calculations received configuration with non-finite values")
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
        if precalced_W is not None:
            W = precalced_W
        elif (cond_tx_x,cond_tx_y) in self.W_dict and not recalculate_W:
            W = self.W_dict[(cond_tx_x,cond_tx_y)]
        else:
            W = self.get_bessel_w(current_parameters, self.device)
            print(W[0][0][1])
            self.W_dict[(cond_tx_x,cond_tx_y)] = W
        H = self.GetH(current_parameters, W)
        return H,W

    def clear_bessel_mem(self):
        self.W_dict = LimitedSizeDict(size_limit=128)
    def generate_tx_location(self,size,device):
        x_tx_orig = torch.tensor([0, 0, 0]).repeat(size, 1).to(device)
        y_tx_orig = torch.tensor([4, 4.5, 5]).repeat(size, 1).to(device)
        tx_x_diff = 19.5 * torch.rand([size, 3], device=device,dtype=torch.float64) - 3.3 # 19.5 *
        tx_y_diff = 11.5 * torch.rand([size, 3], device=device,dtype=torch.float64) - 2.8
        tx_x, tx_y = x_tx_orig + tx_x_diff, y_tx_orig + tx_y_diff
        return tx_x,tx_y
    def change_rx_location(self,x_rx_new,y_rx_new):
        self.parameters["x_rx"] = x_rx_new
        self.parameters["y_rx"] = y_rx_new
        self.clear_bessel_mem()

    def plot_environment(self,tx_x=None,tx_y=None,y_scaler = 1,x_scaler=1,y_shift=0,x_shift=0,show=True):
        plt.scatter(x_scaler*(self.parameters["x_env"]+x_shift),y_scaler*(self.parameters["y_env"]+y_shift))
        if tx_x is not None:
            plt.scatter(tx_x,tx_y)
        if show:
            plt.show()
    def save_and_change_to_clean_environment(self):
        if not self.clean_environment:
            self.clear_bessel_mem()
            self.stored_x_env = self.parameters["x_env"]
            self.stored_y_env = self.parameters["y_env"]
            # plt.scatter(x_env, y_env)
            # plt.show()
            self.parameters["x_env"] = self.parameters["x_env_clean"]
            self.parameters["y_env"] = self.parameters["y_env_clean"]
            self.clean_environment = True

    def reload_original_environment(self):
        if self.clean_environment:
            self.clear_bessel_mem()
            self.parameters["x_env"] = self.stored_x_env
            self.parameters["y_env"] = self.stored_y_env
            self.clean_environment = False

    def set_configuration(self):
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
        self.parameters["x_env"] = torch.tensor(enclosure['x_env']).to(self.device).type(torch.float64)
        self.parameters["y_env"] = torch.tensor(enclosure['y_env']).to(self.device).type(torch.float64)
        self.parameters["x_env_clean"] = torch.tensor(enclosure_clean['x_env']).to(self.device).type(torch.float64)
        self.parameters["y_env_clean"] = torch.tensor(enclosure_clean['y_env']).to(self.device).type(torch.float64)
        self.parameters["fres_env"] = 10 * torch.ones(self.parameters["x_env"].shape).to(self.device).type(torch.float64)
        self.parameters["chi_env"] = 50 * torch.ones(self.parameters["x_env"].shape).to(self.device).type(torch.float64)
        self.parameters["gamma_env"] = 0 * torch.ones(self.parameters["x_env"].shape).to(self.device).type(torch.float64)

        RIS_loc = {}
        scipy.io.loadmat("..//PhysFad//ExampleRIS.mat", RIS_loc)
        self.parameters["x_ris"] = torch.tensor(RIS_loc['x_ris']).to(self.device).type(torch.float64)
        self.parameters["y_ris"] = torch.tensor(RIS_loc['y_ris']).to(self.device).type(torch.float64)
        ris_num_samples = 3
        N_RIS = len(self.parameters["x_ris"][0])
        self.N_RIS = N_RIS
        self.N_RIS_PARAMS = 3*N_RIS
        # logical_fres_ris = torch.reshape(torch.tensor(random.choices([True, False], k=ris_num_samples * N_RIS)), [ris_num_samples, N_RIS])
        # resonant_freq =(1.1-0.9)*torch.rand(ris_num_samples, N_RIS)+0.9
        # non_resonant_freq = (5-1.1)*torch.rand(ris_num_samples, N_RIS)+1.1
        # parameters["fres_ris"] = logical_fres_ris * resonant_freq + (~ logical_fres_ris) * non_resonant_freq
        # chi_ris = 0.2*torch.ones([ris_num_samples,x_ris.shape[1]])
        # gamma_ris = 0*torch.ones([ris_num_samples,x_ris.shape[1]])
        self.parameters["x_ris"] = self.parameters["x_ris"][0].unsqueeze(0).to(self.device).type(torch.float64)
        self.parameters["y_ris"] = self.parameters["y_ris"][0].unsqueeze(0).to(self.device).type(torch.float64)
        RISConfiguration = np.loadtxt("RandomConfiguration.txt")
        self.parameters["fres_ris"] = torch.tensor(RISConfiguration[0:88]).unsqueeze(0).to(self.device).type(torch.float64)
        self.parameters["chi_ris"] = torch.tensor(RISConfiguration[88:176]).unsqueeze(0).to(self.device).type(torch.float64)
        self.parameters["gamma_ris"] = torch.tensor(RISConfiguration[176:264]).unsqueeze(0).to(self.device).type(torch.float64)




        # torch.autograd.set_detect_anomaly(True)
        W = self.get_bessel_w(self.parameters, self.device)
        self.W = W # default W

        return W
    def besselj(self,order,z):
        return 1
    def bessely(self,order,z):
        return torch.special.bessel_y0()
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

        k = 2 * torch.pi * parameters["freq"]
        x = torch.cat([parameters["x_tx"], parameters["x_rx"], parameters["x_env"], parameters["x_ris"]], 1)
        y = torch.cat([parameters["y_tx"], parameters["y_rx"], parameters["y_env"], parameters["y_ris"]], 1)

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
        # print("batched Physfad")
        epsilon = 0.00000001
        k = 2 * torch.pi * parameters["freq"]
        # x = torch.cat([x_tx, x_rx, x_env, x_ris],1)
        # y = torch.cat([y_tx, y_rx, y_env, y_ris],1)
        batch_size = parameters["fres_ris"].shape[0]
        fres = torch.cat([parameters["fres_tx"].repeat(batch_size,1), parameters["fres_rx"].repeat(batch_size,1), parameters["fres_env"].repeat(batch_size,1), parameters["fres_ris"]], 1)
        chi = torch.cat([parameters["chi_tx"].repeat(batch_size,1), parameters["chi_rx"].repeat(batch_size,1), parameters["chi_env"].repeat(batch_size,1), parameters["chi_ris"]], 1)
        gamma = torch.cat([parameters["gamma_tx"].repeat(batch_size,1), parameters["gamma_rx"].repeat(batch_size,1), parameters["gamma_env"].repeat(batch_size,1), parameters["gamma_ris"]], 1)

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
        # if torch.any(~torch.isfinite(inv_alpha)):
        #     print("oh no1")
        W = W_full.clone().repeat(batch_size,1,1,1)
        # width = W.size(0)
        Mask = torch.eye(W.size(2)).repeat(batch_size,len(parameters["freq"]), 1, 1).bool()
        W[Mask] = inv_alpha.permute(0,2,1).reshape(-1)
        W_diag_elem = torch.diagonal(W, dim1=-2, dim2=-1)
        W_diag_matrix = torch.zeros(W.shape, dtype=torch.complex64, device=self.device)
        W_diag_matrix.diagonal(dim1=-2, dim2=-1).copy_(W_diag_elem)
        # if torch.any(~torch.isfinite(W_diag_matrix)):
        #     print("oh no2")
        V = torch.linalg.solve_ex(W, W_diag_matrix)[0]
        # if torch.any(~torch.isfinite(V)):
        #     print("oh no3")
        H = V[:,:, N_T: (N_T + N_R), 0: N_T]
        # if torch.any(~torch.isfinite(H)):
        #     print("oh no4")
        return H

    def GetH(self,parameters, W_full):

        if parameters["fres_ris"].shape[0]!=1:
            return self.GetH_batched(parameters, W_full)
        # print("normal Physfad")
        epsilon = 0.00000001
        k=2*torch.pi*parameters["freq"]
        # x = torch.cat([x_tx, x_rx, x_env, x_ris],1)
        # y = torch.cat([y_tx, y_rx, y_env, y_ris],1)
        fres = torch.cat([parameters["fres_tx"], parameters["fres_rx"], parameters["fres_env"], parameters["fres_ris"]],1)
        chi = torch.cat([parameters["chi_tx"], parameters["chi_rx"], parameters["chi_env"], parameters["chi_ris"]],1)
        gamma = torch.cat([parameters["gamma_tx"], parameters["gamma_rx"], parameters["gamma_env"], parameters["gamma_ris"]],1)

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

        V = torch.linalg.solve(W, W_diag_matrix)
        H = V[:,N_T: (N_T + N_R), 0: N_T]

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
        chi_ris = 0.2 * torch.ones((batch_size, number_of_elements), dtype=torch.float64, device=self.device,requires_grad=False)
        gamma_ris = 0 * torch.zeros((batch_size, number_of_elements), dtype=torch.float64, device=self.device,requires_grad=False)

        return torch.hstack([fres, chi_ris, gamma_ris])

    def scale_output_to_range(self, normelized_output):
        batch_size = normelized_output.shape[0]
        num_config = normelized_output.shape[1]
        assert num_config % 3 == 0, "configuration count should be divisible by three"
        num_ris_elements = num_config // 3
        fres_output = normelized_output[:,0 * num_ris_elements:1 * num_ris_elements]
        chi_output = normelized_output[:,1 * num_ris_elements:2 * num_ris_elements]
        gamma_output = normelized_output[:,2 * num_ris_elements:3 * num_ris_elements]
        fres_output_rescaled = fres_output * self.config.fres_max_range
        chi_output_rescaled = chi_output * self.config.chi_max_range
        gamma_output_rescaled = gamma_output * self.config.gamma_max_range
        return torch.hstack([fres_output_rescaled, chi_output_rescaled, gamma_output_rescaled])

