import torch
import scipy.special as scp
import numpy as np
import scipy.io
import math
import datetime
from utils import OrderedDict
class physfad_c():
    def __init__(self,config,device):
        self.config = config
        self.device = device
        self.W_dict = OrderedDict(size_limit=256)

    def __call__(self,ris_configuration_normalized,cond_tx_x,cond_tx_y):
        (freq, x_tx, y_tx, fres_tx, chi_tx, gamma_tx,
         x_rx, y_rx, fres_rx, chi_rx, gamma_rx,
         x_env, y_env, fres_env, chi_env, gamma_env, x_ris_c, y_ris_c) = self.parameters
        if torch.any(~torch.isfinite(ris_configuration_normalized)):
            print("in physfad H calculations received configuration with non-finite values")
        if not torch.is_tensor(ris_configuration_normalized):
            ris_configuration_normalized = torch.tensor(ris_configuration_normalized, device=self.device)

        ris_configuration_full = self.fill_ris_config(ris_configuration_normalized)
        # ris_configuration = ris_configuration_full
        ris_configuration = self.scale_output_to_range(ris_configuration_full)
        # fres_ris_c  = ris_configuration[:, 0:45]
        # chi_ris_c   = ris_configuration[:, 45:90]
        # gamma_ris_c = ris_configuration[:, 90:135]
        fres_ris_c  = ris_configuration[:, self.N_RIS*0:self.N_RIS*1]
        chi_ris_c   = ris_configuration[:, self.N_RIS*1:self.N_RIS*2]
        gamma_ris_c = ris_configuration[:, self.N_RIS*2:self.N_RIS*3]
        # if (cond_tx_x,cond_tx_y) in self.W_dict:
        #     W = self.W_dict[(cond_tx_x,cond_tx_y)]
        # else:
        W = self.get_bessel_w(self.freq,
                         cond_tx_x, cond_tx_y,
                         x_rx, y_rx,
                         x_env, y_env,
                         x_ris_c, y_ris_c, self.device)
        self.W_dict[(cond_tx_x,cond_tx_y)] = W
        H = self.GetH(freq, W,
                 cond_tx_x, cond_tx_y, fres_tx, chi_tx, gamma_tx,
                 x_rx, y_rx, fres_rx, chi_rx, gamma_rx,
                 x_env, y_env, fres_env, chi_env, gamma_env,
                 x_ris_c, y_ris_c, fres_ris_c, chi_ris_c, gamma_ris_c)
        return H
    def set_parameters(self):
        '''
            set the basic ris configuration and pack it into one tuple: parameters
        '''
        freq = torch.tensor(np.linspace(0.9, 1.1, 120));

        ## Configurable Dipole Properties
        ## Transmitters ##
        # locations
        x_tx = torch.tensor([0, 0, 0],dtype=torch.float64).unsqueeze(0).to(self.device)
        y_tx = torch.tensor([4, 4.5, 5],dtype=torch.float64).unsqueeze(0).to(self.device)
        # dipole properties
        fres_tx = torch.tensor([1, 1, 1],dtype=torch.float64).unsqueeze(0).to(self.device)
        chi_tx = torch.tensor([0.5, 0.5, 0.5],dtype=torch.float64).unsqueeze(0).to(self.device)
        gamma_tx = torch.tensor([0, 0, 0],dtype=torch.float64).unsqueeze(0).to(self.device)

        ##  Receivers ##
        # locations
        x_rx = torch.tensor([15, 15, 15, 15],dtype=torch.float64).unsqueeze(0).to(self.device)
        y_rx = torch.tensor([11, 11.5, 12, 12.5],dtype=torch.float64).unsqueeze(0).to(self.device)
        # properties
        fres_rx = torch.tensor([1, 1, 1, 1],dtype=torch.float64).unsqueeze(0).to(self.device)
        chi_rx = torch.tensor([0.5, 0.5, 0.5, 0.5],dtype=torch.float64).unsqueeze(0).to(self.device)
        gamma_rx = torch.tensor([0, 0, 0, 0],dtype=torch.float64).unsqueeze(0).to(self.device)

        enclosure = {}
        scipy.io.loadmat("..//PhysFad//ComplexEnclosure2.mat", enclosure)
        x_env = torch.tensor(enclosure['x_env']).to(self.device).type(torch.float64)
        y_env = torch.tensor(enclosure['y_env']).to(self.device).type(torch.float64)
        fres_env = 10 * torch.ones(x_env.shape).to(self.device).type(torch.float64)
        chi_env = 50 * torch.ones(x_env.shape).to(self.device).type(torch.float64)
        gamma_env = 0 * torch.ones(x_env.shape).to(self.device).type(torch.float64)

        RIS_loc = {}
        scipy.io.loadmat("..//PhysFad//ExampleRIS.mat", RIS_loc)
        x_ris = torch.tensor(RIS_loc['x_ris']).to(self.device).type(torch.float64)
        y_ris = torch.tensor(RIS_loc['y_ris']).to(self.device).type(torch.float64)
        x_ris_c = x_ris[0].unsqueeze(0).to(self.device).type(torch.float64)
        y_ris_c = y_ris[0].unsqueeze(0).to(self.device).type(torch.float64)
        parameters = (freq, x_tx, y_tx, fres_tx, chi_tx, gamma_tx,
                      x_rx, y_rx, fres_rx, chi_rx, gamma_rx,
                      x_env, y_env, fres_env, chi_env, gamma_env, x_ris_c, y_ris_c)
        self.parameters = parameters
        return parameters
    def set_configuration(self):
        self.freq = torch.tensor(np.linspace(0.9, 1.1, 120));

        ## Configurable Dipole Properties
        ## Transmitters ##
        # locations
        x_tx = torch.tensor([0, 0, 0]).unsqueeze(0).to(self.device).type(torch.float64)
        y_tx = torch.tensor([4, 4.5, 5]).unsqueeze(0).to(self.device).type(torch.float64)
        # dipole properties
        fres_tx = torch.tensor([1, 1, 1]).unsqueeze(0).to(self.device).type(torch.float64)
        chi_tx = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(0).to(self.device).type(torch.float64)
        gamma_tx = torch.tensor([0, 0, 0]).unsqueeze(0).to(self.device).type(torch.float64)

        ##  Receivers ##
        # locations
        x_rx = torch.tensor([15, 15, 15, 15]).unsqueeze(0).to(self.device).type(torch.float64)
        y_rx = torch.tensor([11, 11.5, 12, 12.5]).unsqueeze(0).to(self.device).type(torch.float64)
        # properties
        fres_rx = torch.tensor([1, 1, 1, 1]).unsqueeze(0).to(self.device).type(torch.float64)
        chi_rx = torch.tensor([0.5, 0.5, 0.5, 0.5]).unsqueeze(0).to(self.device).type(torch.float64)
        gamma_rx = torch.tensor([0, 0, 0, 0]).unsqueeze(0).to(self.device).type(torch.float64)

        enclosure = {}
        scipy.io.loadmat("..//PhysFad//ComplexEnclosure2.mat", enclosure)
        x_env = torch.tensor(enclosure['x_env']).to(self.device).type(torch.float64)
        y_env = torch.tensor(enclosure['y_env']).to(self.device).type(torch.float64)
        fres_env = 10 * torch.ones(x_env.shape).to(self.device).type(torch.float64)
        chi_env = 50 * torch.ones(x_env.shape).to(self.device).type(torch.float64)
        gamma_env = 0 * torch.ones(x_env.shape).to(self.device).type(torch.float64)

        RIS_loc = {}
        scipy.io.loadmat("..//PhysFad//ExampleRIS.mat", RIS_loc)
        x_ris = torch.tensor(RIS_loc['x_ris']).to(self.device).type(torch.float64)
        y_ris = torch.tensor(RIS_loc['y_ris']).to(self.device).type(torch.float64)
        ris_num_samples = 3
        N_RIS = len(x_ris[0])
        self.N_RIS = N_RIS
        self.N_RIS_PARAMS = 3*N_RIS
        # logical_fres_ris = torch.reshape(torch.tensor(random.choices([True, False], k=ris_num_samples * N_RIS)), [ris_num_samples, N_RIS])
        # resonant_freq =(1.1-0.9)*torch.rand(ris_num_samples, N_RIS)+0.9
        # non_resonant_freq = (5-1.1)*torch.rand(ris_num_samples, N_RIS)+1.1
        # fres_ris = logical_fres_ris * resonant_freq + (~ logical_fres_ris) * non_resonant_freq
        # chi_ris = 0.2*torch.ones([ris_num_samples,x_ris.shape[1]])
        # gamma_ris = 0*torch.ones([ris_num_samples,x_ris.shape[1]])
        x_ris_c = x_ris[0].unsqueeze(0).to(self.device).type(torch.float64)
        y_ris_c = y_ris[0].unsqueeze(0).to(self.device).type(torch.float64)
        RISConfiguration = np.loadtxt("RandomConfiguration.txt")
        fres_ris_c = torch.tensor(RISConfiguration[0:88]).unsqueeze(0).to(self.device).type(torch.float64)
        chi_ris_c = torch.tensor(RISConfiguration[88:176]).unsqueeze(0).to(self.device).type(torch.float64)
        gamma_ris_c = torch.tensor(RISConfiguration[176:264]).unsqueeze(0).to(self.device).type(torch.float64)

        # RisConfiguration = torch.hstack([fres_ris_c,chi_ris_c,gamma_ris_c])

        # H = GetH(freq,
        #      x_tx,y_tx,fres_tx,chi_tx,gamma_tx,
        #      x_rx,y_rx,fres_rx,chi_rx,gamma_rx,
        #      x_env,y_env,fres_env,chi_env,gamma_env,
        #      x_ris_c,y_ris_c,fres_ris_c,chi_ris_c,gamma_ris_c)

        parameters = (self.freq, x_tx, y_tx, fres_tx, chi_tx, gamma_tx,
                      x_rx, y_rx, fres_rx, chi_rx, gamma_rx,
                      x_env, y_env, fres_env, chi_env, gamma_env, x_ris_c, y_ris_c)
        # torch.autograd.set_detect_anomaly(True)
        W = self.get_bessel_w(self.freq,
                         x_tx, y_tx,
                         x_rx, y_rx,
                         x_env, y_env,
                         x_ris_c, y_ris_c, self.device)
        self.W = W
        self.parameters = parameters
        return parameters, W
    def besselj(self,order,z):
        return 1
    def bessely(self,order,z):
        return torch.special.bessel_y0()
    def besselh(self,order,kind=2,z=0,scale=0):
        # return besselj(0,z)-torch.tensor([1j])*bessely(0,z);
        return scp.hankel2(order, z)
    def get_bessel_w(self,freq,
             x_tx,y_tx,
             x_rx,y_rx,
             x_env,y_env,
             x_ris,y_ris,device):
        k = 2 * torch.pi * freq
        x = torch.cat([x_tx, x_rx, x_env, x_ris], 1)
        y = torch.cat([y_tx, y_rx, y_env, y_ris], 1)

        N_T = len(x_tx[0])
        N_R = len(x_rx[0])
        N_E = len(x_env[0])
        N_RIS = len(x_ris[0])
        N = N_T + N_R + N_E + N_RIS
        H = torch.zeros([len(freq), N_R, N_T], dtype=torch.complex64, device=device)
        pi = torch.pi
        W = torch.zeros([len(freq),N,N],dtype=torch.complex64,device=device)
        for f in range(len(freq)):
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
    def GetH_batched(self,freq, W_full,
             x_tx,y_tx,fres_tx,chi_tx,gamma_tx,
             x_rx,y_rx,fres_rx,chi_rx,gamma_rx,
             x_env,y_env,fres_env,chi_env,gamma_env,
             x_ris,y_ris,fres_ris,chi_ris,gamma_ris):
        # print("batched Physfad")
        epsilon = 0.00000001
        k = 2 * torch.pi * freq
        # x = torch.cat([x_tx, x_rx, x_env, x_ris],1)
        # y = torch.cat([y_tx, y_rx, y_env, y_ris],1)
        batch_size = fres_ris.shape[0]
        fres = torch.cat([fres_tx.repeat(batch_size,1), fres_rx.repeat(batch_size,1), fres_env.repeat(batch_size,1), fres_ris], 1)
        chi = torch.cat([chi_tx.repeat(batch_size,1), chi_rx.repeat(batch_size,1), chi_env.repeat(batch_size,1), chi_ris], 1)
        gamma = torch.cat([gamma_tx.repeat(batch_size,1), gamma_rx.repeat(batch_size,1), gamma_env.repeat(batch_size,1), gamma_ris], 1)

        N_T = len(x_tx[0])
        N_R = len(x_rx[0])
        N_E = len(x_env[0])
        N_RIS = len(x_ris[0])
        N = N_T + N_R + N_E + N_RIS
        pi = torch.pi
        k2 = (torch.pow(k, 2)).repeat(batch_size,1).to(self.device)
        two_pi = 2 * pi
        two_pi_freq = (two_pi * freq).repeat(batch_size,1).to(self.device)
        two_pi_freq2 = torch.pow(two_pi_freq, 2)

        chi2 = torch.pow(chi, 2)+epsilon
        two_pi_fres2 = torch.pow((two_pi * fres), 2)

        inv_alpha = (two_pi_fres2.unsqueeze(2) - two_pi_freq2.unsqueeze(1)) / (chi2.unsqueeze(2)) + 1j * ((
                    k2.unsqueeze(1) / 4) + two_pi_freq.unsqueeze(1) * gamma.unsqueeze(2) / chi2.unsqueeze(2))
        inv_alpha = inv_alpha.type(torch.complex64)
        if torch.any(~torch.isfinite(inv_alpha)):
            print("oh no1")
        W = W_full.clone().repeat(batch_size,1,1,1)
        # width = W.size(0)
        Mask = torch.eye(W.size(2)).repeat(batch_size,len(freq), 1, 1).bool()
        W[Mask] = inv_alpha.permute(0,2,1).reshape(-1)
        W_diag_elem = torch.diagonal(W, dim1=-2, dim2=-1)
        W_diag_matrix = torch.zeros(W.shape, dtype=torch.complex64, device=self.device)
        W_diag_matrix.diagonal(dim1=-2, dim2=-1).copy_(W_diag_elem)
        if torch.any(~torch.isfinite(W_diag_matrix)):
            print("oh no2")
        V = torch.linalg.solve(W, W_diag_matrix)
        if torch.any(~torch.isfinite(V)):
            print("oh no3")
        H = V[:,:, N_T: (N_T + N_R), 0: N_T]
        if torch.any(~torch.isfinite(H)):
            print("oh no4")
        return H

    def GetH(self,freq, W_full,
             x_tx,y_tx,fres_tx,chi_tx,gamma_tx,
             x_rx,y_rx,fres_rx,chi_rx,gamma_rx,
             x_env,y_env,fres_env,chi_env,gamma_env,
             x_ris,y_ris,fres_ris,chi_ris,gamma_ris):

        if fres_ris.shape[0]!=1:
            return self.GetH_batched(freq, W_full,
                 x_tx,y_tx,fres_tx,chi_tx,gamma_tx,
                 x_rx,y_rx,fres_rx,chi_rx,gamma_rx,
                 x_env,y_env,fres_env,chi_env,gamma_env,
                 x_ris,y_ris,fres_ris,chi_ris,gamma_ris)
        # print("normal Physfad")
        epsilon = 0.00000001
        k=2*torch.pi*freq
        # x = torch.cat([x_tx, x_rx, x_env, x_ris],1)
        # y = torch.cat([y_tx, y_rx, y_env, y_ris],1)
        fres = torch.cat([fres_tx, fres_rx, fres_env, fres_ris],1)
        chi = torch.cat([chi_tx, chi_rx, chi_env, chi_ris],1)
        gamma = torch.cat([gamma_tx, gamma_rx, gamma_env, gamma_ris],1)

        N_T   = len(x_tx[0])
        N_R   = len(x_rx[0])
        N_E   = len(x_env[0])
        N_RIS = len(x_ris[0])
        N = N_T + N_R + N_E + N_RIS
        H = torch.zeros([len(freq),N_R,N_T],dtype=torch.complex64,device=self.device)
        pi = torch.pi
        k2 = (torch.pow(k, 2)).to(self.device)
        two_pi = 2 * pi
        two_pi_freq = (two_pi * freq).to(self.device)
        two_pi_freq2 = torch.pow(two_pi_freq, 2)
        chi2 = torch.pow(chi[0, :], 2)+epsilon
        gamma_ = gamma[0,:]
        two_pi_fres2 = torch.pow((two_pi * fres[0, :]), 2)

        inv_alpha = (two_pi_fres2.unsqueeze(1) - two_pi_freq2.unsqueeze(0)) / (chi2.unsqueeze(1)) + 1j * ((k2.unsqueeze(0) / 4) + two_pi_freq.unsqueeze(0) * gamma_.unsqueeze(1) / chi2.unsqueeze(1))
        inv_alpha = inv_alpha.type(torch.complex64)
        W = W_full.clone()
        # width = W.size(0)
        Mask = torch.eye(W.size(1)).repeat(len(freq), 1, 1).bool()
        W[Mask] = inv_alpha.T.reshape(-1)
        W_diag_elem = torch.diagonal(W,dim1=-2,dim2=-1)
        W_diag_matrix = torch.zeros(W.shape,dtype=torch.complex64,device=self.device)
        W_diag_matrix.diagonal(dim1=-2,dim2=-1).copy_(W_diag_elem)
        # scipy.io.savemat("W_mat.mat", {"W_mat": W.cpu().detach().numpy()})

        V = torch.linalg.solve(W, W_diag_matrix)
        H = V[:,N_T: (N_T + N_R), 0: N_T]
        if torch.any(~torch.isfinite(H)):
            print("physfad(not batched) nan detected")

        # for f in range(len(freq)):
            # x_diff = torch.zeros([N,N],dtype=torch.float64,device=device)
            # y_diff = torch.zeros([N, N],dtype=torch.float64,device=device)
            # for l in range(N):
            #     xl_vec = x[0,l] * torch.ones([1, N],dtype=torch.float64,device=device)
            #     yl_vec = y[0,l] * torch.ones([1, N],dtype=torch.float64,device=device)
            #     x_diff[l,:] = x - xl_vec
            #     y_diff[l,:] = y - yl_vec
            # BesselInp = k[f]*torch.sqrt(x_diff**2+y_diff**2)
            # BesselOut = torch.tensor(besselh(0,2,BesselInp.cpu().numpy()),device=device)
            # W = 1j * (k[f] ** 2 / 4) * BesselOut
            # W = W_full[f].clone()

            # start_time = datetime.datetime.now()
            # inv_alpha = (two_pi_fres2 - two_pi_freq2[f]) / (chi2) + 1j * (k2[f] / 4) + two_pi_freq[f] * gamma_ / chi2
            # width = W.size(0)
            # W.as_strided([width], [width + 1]).copy_(inv_alpha[:,f])
            # W.as_strided([width], [width + 1]).copy_(inv_alpha)
            # for i in range(0,N):
                # diagonal entries of W are the inverse polarizabilities
                # a =
                # b =

                # inv_alpha = a
                # W[i, i] = inv_alpha[i]
                # if i==(N-1):
                #     scipy.io.savemat("inv_alpha.mat", {"inv_alpha": inv_alpha.cpu().detach().numpy()})



            #%Invert W and extract H
            # np.savetxt("W_mat.txt", W.cpu().detach().numpy())
            # scipy.io.savemat("W_mat.mat", {"W_mat": W.cpu().detach().numpy()})
            # Winv = torch.inverse(W)
            # V = torch.matmul(torch.diag(torch.diag(W)),Winv)
            # after_mat_calc = datetime.datetime.now()
            # calc_time = after_mat_calc - start_time
            # print("matrix calculation time {0}".format(str(calc_time)))
            # V = torch.linalg.solve(W, torch.diag(torch.diag(W)))
            # V = torch.linalg.solve(W[f], torch.diag(torch.diag(W[f])))
            # scipy.io.savemat("W_mat.mat", {"W_mat": W})
            # inv_time = datetime.datetime.now() - after_mat_calc
            # print("matrix inversion time {0}".format(str(inv_time)))
            # H[f,:,:] = V[N_T: (N_T + N_R), 0: N_T]

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

#             ((2*pi*fres(ii))^2-(2*pi*freq(ff))^2)/
#             ((2 * torch.pi * fres[0, i]) ** 2 - (2 * torch.pi * freq[f]) ** 2)
#
#             (chi(ii)^2) + 1i*(((k(ff)^2)/4) + 2*pi*freq(ff)*gamma(ii)/(chi(ii)^2));
#             (chi[0,i] ** 2) + 1j * (((k[f] ** 2) / 4) + 2 * torch.pi * freq[f] * gamma[0,i] / (chi[0,i] ** 2))
#
# inv_alpha = ((2 * torch.pi * fres[0,i]) ** 2 - (2 * torch.pi * freq[f]) ** 2)/
#              (chi[0,i] ** 2) + 1j * ((k[f] ** 2 / 4) + 2 * torch.pi * freq[f] * gamma[0,i] / (chi[0,i] ** 2))