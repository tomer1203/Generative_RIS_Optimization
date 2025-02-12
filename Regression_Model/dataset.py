import torch as T
from sklearn.decomposition import PCA
import numpy as np

import scipy.io
## DEBUG!!
from ChannelMatrixEvaluation import test_configurations_capacity
import torch
## DEBUG!!
from utils import get_physfad_grads,copy_with_gradients
from rate_model import capacity_loss
class RISDataset(T.utils.data.Dataset):

    def __init__(self, RIS_config_path, H_realizations_file, H_capacity_path, RIS_Gradients_path,tx_location_path,batch_size, virtual_batch_size=256, output_size =120, output_shape =(4, 3), calculate_capacity = False,calculate_gradients=False, physfad=None, only_fres=False, m_rows=None, device=T.device("cpu")):
        # ris_configs_np = np.loadtxt(RIS_config_file,delimiter=",", dtype=np.float32)
        # H_realiz_np = np.loadtxt(H_realizations_file,delimiter=",", dtype=np.float32)
        enclosure = {}
        scipy.io.loadmat(RIS_config_path, enclosure)
        ris_configs_np = enclosure["RISConfiguration"]
        enclosure = {}
        scipy.io.loadmat(H_realizations_file,enclosure)
        H_realiz_np = enclosure["sampled_Hs"].reshape(-1,output_size*output_shape[0]*output_shape[1])

        scipy.io.loadmat(tx_location_path+"conditional_transmitter_x_location.mat", enclosure)
        tx_x_location = enclosure["x_tx_modified"]
        scipy.io.loadmat(tx_location_path + "conditional_transmitter_y_location.mat", enclosure)
        tx_y_location = enclosure["y_tx_modified"]
        ## DEBUG!!
        # physfad_capacity, physfad_H = test_configurations_capacity(get_configuration_parameters(device),
        #                                                            torch.Tensor(ris_configs_np[0, :]).unsqueeze(0).to(device),
        #                                                            device)
        ## DEBUG!!
        # self.pca = PCA(n_components=20)
        # self.pca.fit(H_realiz_np)
        # self.reduced_dimensionality_realizations = self.pca.fit_transform(H_realiz_np)
        if only_fres:
            self.x_data = T.tensor(ris_configs_np[:,0:45], dtype=T.float32).to(device)
        else:
            self.x_data = T.tensor(ris_configs_np, dtype=T.float32).to(device)
        self.y_data = T.tensor(H_realiz_np, dtype=T.complex64).to(device)
        self.tx_x = T.tensor(tx_x_location,dtype=T.float32,device=device)
        self.tx_y = T.tensor(tx_y_location,dtype=T.float32,device=device)
        if calculate_capacity:
            print("calculating capacity")
            self.y_capacity = capacity_loss(self.y_data.reshape((-1,output_size,output_shape[0],output_shape[1])),list_out=True,device=device)
            np.savetxt(H_capacity_path, self.y_capacity.cpu().detach().numpy(), delimiter=",")
            print("Done calculating")
        else:
            self.y_capacity = T.tensor(np.loadtxt(H_capacity_path, delimiter=",", dtype=np.float32), dtype=T.float32, device=device)
        self.device = device
        self.dataset_changed = False
        self.gradients_path = RIS_Gradients_path
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        if calculate_gradients:
            self.gradients = self.calc_gradients(physfad)
        else:
            self.gradients = torch.load(self.gradients_path,).to(self.device)

        # self.y_data = T.tensor(self.reduced_dimensionality_realizations, \
        #                        dtype=T.float32).to(device)
    def calc_gradients(self,physfad,noise=None):
        # self.gradients = torch.zeros(self.x_data.shape,device=self.device)
        # for idx,x in enumerate(self.x_data):
        #     print(idx)
        #     x_with_grads = x.clone().detach().requires_grad_(True).to(self.device)
        #     self.gradients[idx] = get_physfad_grads(x_with_grads,physfad,rate_model,noise=None)
        x_with_grads = copy_with_gradients(self.x_data)
        self.gradients = torch.zeros_like(x_with_grads)
        for i in range(len(self.tx_x)):
            current_tx_x = self.tx_x[i].unsqueeze(0)
            current_tx_y = self.tx_y[i].unsqueeze(0)
            current_x = x_with_grads[i*self.virtual_batch_size:(i+1)*self.virtual_batch_size]
            self.gradients[i*self.virtual_batch_size:(i+1)*self.virtual_batch_size] = \
                get_physfad_grads(current_x, current_tx_x,current_tx_y, physfad, device=self.device,noise=noise)

        torch.save(self.gradients,self.gradients_path)
        return self.gradients
    def add_new_items(self,X,X_gradients,Y,Y_capacity):
        if X is None:
            return
        if torch.any(~torch.isfinite(X)):
            print("non finite value detected")
        self.x_data = T.vstack([self.x_data,X])
        self.gradients = T.vstack([self.gradients,X_gradients])
        self.y_data = T.vstack([self.y_data,Y])
        self.y_capacity = T.hstack([self.y_capacity,Y_capacity])
        self.dataset_changed = True
    def save_dataset(self,RIS_config_file,gradients_file,H_realiz_file,capacity_file):
        if capacity_file is not None:
            np.savetxt(capacity_file, self.y_capacity.cpu().detach().numpy(), delimiter=",")
        if gradients_file is not None:
            torch.save(self.gradients,gradients_file)
        if RIS_config_file is not None:
            scipy.io.savemat(RIS_config_file, {"RISConfiguration": self.x_data.cpu().detach().numpy()})

        if H_realiz_file is not None:
            scipy.io.savemat(H_realiz_file, {"sampled_Hs": self.y_data.cpu().detach().numpy()})

            # ## DEBUG
            # enclosure = {}
            # scipy.io.loadmat(RIS_config_file, enclosure)
            # ris_config = enclosure["RISConfiguration"]
            # print(ris_config)
            # if not torch.equal(self.x_data,T.tensor(ris_config, \
            #                                         dtype=T.float32).to("cpu")):
            #     print("data not equal after save")
            # train_ds_debug = T.tensor(H_realiz_np, dtype=T.float32).to("cpu")
            # if torch.any(~torch.isfinite(train_ds_debug)):
            #     print("non finite value detected")
            # train_ldr_debug = T.utils.data.DataLoader(train_ds_debug, batch_size=32, shuffle=True)
            # if torch.any(~torch.isfinite(train_ldr_debug)):
            #     print("non finite value detected")
            #
            # ## End DEBUG

    def __len__(self):
        return len(self.x_data)//self.virtual_batch_size

    def __getitem__(self, idx):
        preds = self.x_data[self.batch_size*idx:self.batch_size*(idx+1), :]  # or just [idx]
        tx_x = self.tx_x[int(idx//(self.virtual_batch_size/self.batch_size)),:]#(idx*batch_size)/256
        tx_y = self.tx_y[int(idx//(self.virtual_batch_size/self.batch_size)),:]
        gt_capacity = self.y_capacity[self.batch_size*idx:self.batch_size*(idx+1)]
        gt_gradients = self.gradients[self.batch_size*idx:self.batch_size*(idx+1)]
        gt = self.y_data[self.batch_size*idx:self.batch_size*(idx+1), :]

        return (preds,gt_gradients,tx_x,tx_y, gt_capacity, gt)  # tuple of two matrices
