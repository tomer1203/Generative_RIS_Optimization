import torch
import torch as T

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import matplotlib.pyplot as plt
# layer_shapes = [
#     [(64, 100), (64,)],  # (w0, b0)
#     [(10, 64), (10,)]  # (w1, b1)
# ]

def get_layer_shapes(layer_sizes):
    layer_shapes = []
    total_weights_num = 0
    for i,layer_size in enumerate(layer_sizes):
        if i == 0:
            continue
        layer_input_size = layer_sizes[i-1]
        layer_output_size = layer_sizes[i]
        # Shape breakdown: W*XT+b  W=(out,in) * XT= (in,batch) + batch=(out,)
        layer_weight_shape = (layer_output_size,layer_input_size)
        layer_bias_shape = (layer_output_size,)
        layer_shapes.append([layer_weight_shape,layer_bias_shape])
        total_weights_num += np.prod(layer_weight_shape)+layer_bias_shape[0]
    return layer_shapes,total_weights_num
def get_layer_shapes_row_hypernet(layer_sizes):
    layer_shapes = []
    total_weights_num = 0
    for i,layer_size in enumerate(layer_sizes):
        if i == 0:
            continue
        layer_input_size = layer_sizes[i-1]
        layer_output_size = layer_sizes[i]
        # Shape breakdown: W*XT+b  W=(out,in) * XT= (in,batch) + batch=(out,)
        layer_weight_shape = (layer_output_size,)
        layer_bias_shape = (layer_output_size,)
        layer_shapes.append([layer_weight_shape,layer_bias_shape])
        total_weights_num += layer_weight_shape[0]+layer_bias_shape[0]
    return layer_shapes,total_weights_num
class hypernetwork_and_main_model(nn.Module):
    def __init__(self,input_size,hidden_size,output_length,output_shape,model_output_capacity):
        super(hypernetwork_and_main_model, self).__init__()
        # calculate the product of a list
        prod = lambda lst: reduce(lambda x, y: x * y, lst)

        self.model_output_capacity = model_output_capacity
        # setting the layer sizes
        if self.model_output_capacity:
            layer_sizes = [input_size, hidden_size * 2, hidden_size, hidden_size // 2, hidden_size // 3, 1]
        else:
            layer_sizes = [input_size, hidden_size * 6, hidden_size*4, hidden_size * 3,
                           hidden_size * 8, hidden_size * 16,output_length * prod(output_shape)]


        self.layer_shapes,self.total_weights_num = get_layer_shapes_row_hypernet(layer_sizes)
        self.hyp_net = hypernetwork(self.total_weights_num)
        self.main_net = main_Net(layer_sizes,model_output_capacity)
        print("printing layer sizes")
        for name, param in self.main_net.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} \n")
    def forward(self,tx_x,tx_y,x):
        weights = self.hyp_net(tx_x,tx_y) # get the weights of the main network(gain and shift)
        output = self.main_net(x, self.layer_shapes, weights)
        return output
    def to(self,*args,**kwargs):
        self.hyp_net.to(*args,**kwargs)
        self.main_net.to(*args,**kwargs)
        return self





class hypernetwork(nn.Module):
    def __init__(self,total_weights_num):
        super(hypernetwork, self).__init__()
        self.hid1 = nn.Linear(6,150)
        self.hid2 = nn.Linear(150,500)
        self.hid3 = nn.Linear(500,total_weights_num)
        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
    def forward(self,tx_x,tx_y):
        input_location = T.hstack([tx_x,tx_y])
        z = F.relu(self.hid1(input_location))
        z = F.relu(self.hid2(z))
        z = F.relu(self.hid3(z))
        return z
class main_Net(nn.Module):
    def __init__(self,layer_sizes,model_output_capacity):
        super(main_Net, self).__init__()

        self.layer_sizes = layer_sizes
        self.linear_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i,size in enumerate(layer_sizes):
            if i == 0:
                last_size = size
                continue
            self.linear_layers.append(nn.Linear(last_size,size))
            self.batchnorm_layers.append(nn.BatchNorm1d(size))
            self.dropout_layers.append(nn.Dropout(0.04))
            nn.init.xavier_uniform_(self.linear_layers[-1].weight)
            nn.init.zeros_(self.linear_layers[-1].bias)
            last_size = size
    def reshape_weights(self,layer_shapes,pred_weights):
        idx, params = 0, []
        for layer in layer_shapes:
            layer_params = []
            for shape in layer:
                offset = np.prod(shape)
                layer_params.append(pred_weights[:, idx: idx + offset].reshape(shape))
                idx += offset
            params.append(layer_params)
        return params
    def reshape_weights_linear_hypernet(self,layer_shapes,pred_weights):
        idx, params = 0, []
        for layer in layer_shapes:
            layer_params = []
            for shape in layer:
                offset = shape[0]
                layer_params.append(pred_weights[:, idx: idx + offset].reshape(shape))
                idx += offset
            params.append(layer_params)
        return params
    def forward(self, x,layer_shapes,predicted_flat_weights):
        hypernetwork_weights = self.reshape_weights(layer_shapes,predicted_flat_weights)
        z = x**2 # make sure that all configurations are positive
        for i,(gain,shift) in enumerate(hypernetwork_weights):
            z = T.relu(self.linear_layers[i](z)*gain+shift)
            z = self.dropout_layers[i](z)
            # plt.plot(z[0].cpu().detach().numpy())
            # print(z[0])


        return z
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform(m.weight)
        m.bias.data.fill_(0)

# Old network(no use of hypernetwork)
class Net_diffusion(nn.Module):
    def __init__(self,config):
        super(Net_diffusion, self).__init__()
        # calculate the product of a list
        prod = lambda lst: reduce(lambda x, y: x * y, lst)

        input_size  = config.diffusion_inp_size
        output_size = config.input_size # the diffusion network is a denoising network
        hidden_size = config.hidden_size

        self.hid1 = nn.Linear(input_size, hidden_size * 6,dtype=torch.float64)  # 8-(10-10)-1
        self.dropout1 = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(hidden_size * 6,dtype=torch.float64)
        self.hid2 = nn.Linear(6 * hidden_size, hidden_size * 4,dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(hidden_size * 4,dtype=torch.float64)
        self.hid3 = nn.Linear(4 * hidden_size, hidden_size * 3,dtype=torch.float64)
        self.bn3 = nn.BatchNorm1d(hidden_size * 3,dtype=torch.float64)
        self.hid4 = nn.Linear(3 * hidden_size, 5 * hidden_size,dtype=torch.float64)
        self.dropout2 = nn.Dropout(0.1)
        self.hid5 = nn.Linear(5 * hidden_size, 8 * hidden_size,dtype=torch.float64)
        self.bn4 = nn.BatchNorm1d(8 * hidden_size,dtype=torch.float64)
        self.oupt = nn.Linear(8*hidden_size, output_size,dtype=torch.float64)
        self.hid1.apply(init_weights)
        self.hid2.apply(init_weights)
        self.hid3.apply(init_weights)
        self.hid4.apply(init_weights)
        self.hid5.apply(init_weights)
        # self.oupt.apply(init_weights)
        # nn.init.xavier_uniform_(self.hid1.weight)
        # nn.init.zeros_(self.hid1.bias)
        # nn.init.xavier_uniform_(self.hid2.weight)
        # nn.init.zeros_(self.hid2.bias)
        # nn.init.xavier_uniform_(self.hid3.weight)
        # nn.init.zeros_(self.hid3.bias)
        # nn.init.xavier_uniform_(self.hid4.weight)
        # nn.init.zeros_(self.hid4.bias)
        # nn.init.xavier_uniform_(self.hid5.weight)
        # nn.init.zeros_(self.hid5.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

    def forward(self, x):

        z = T.relu(self.hid1(x))
        # if x.shape[0] != 1:  # batch_size==1
        z = self.bn1(z)
        # z = self.dropout1(z)
        z = T.relu(self.hid2(z))
        z = self.bn2(z)
        # z = self.bn2(z)
        z = T.relu(self.hid3(z))
        z = self.bn3(z)
        z = T.relu(self.hid4(z))
        z = T.relu(self.hid5(z))
        z = self.bn4(z)
        # z = self.dropout2(z)
        z = self.oupt(z)  # no activation

        normalized_output = T.nn.functional.sigmoid(z)
        return normalized_output
class Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_length,output_shape,model_output_capacity):
        super(Net, self).__init__()
        # calculate the product of a list
        prod = lambda lst: reduce(lambda x, y: x * y, lst)


        if model_output_capacity:
            self.hid1 = nn.Linear(input_size, hidden_size * 3)  # 8-(10-10)-1
            self.dropout1 = nn.Dropout(0.1)
            self.bn1 = nn.BatchNorm1d(hidden_size * 3)
            self.hid2 = nn.Linear(3 * hidden_size, hidden_size * 2)
            self.bn2 = nn.BatchNorm1d(hidden_size * 2)
            self.hid3 = nn.Linear(2 * hidden_size, hidden_size)
            self.bn3 = nn.BatchNorm1d(hidden_size)
            self.hid4 = nn.Linear(hidden_size, hidden_size//2)
            self.dropout2 = nn.Dropout(0.1)
            self.hid5 = nn.Linear(hidden_size//2, hidden_size//3)
            self.oupt = nn.Linear(hidden_size//3, 1)
        else:
            self.hid1 = nn.Linear(input_size, hidden_size * 6)  # 8-(10-10)-1
            self.dropout1 = nn.Dropout(0.1)
            self.bn1 = nn.BatchNorm1d(hidden_size * 6)
            self.hid2 = nn.Linear(6 * hidden_size, hidden_size * 4)
            self.bn2 = nn.BatchNorm1d(hidden_size * 4)
            self.hid3 = nn.Linear(4 * hidden_size, hidden_size * 3)
            self.bn3 = nn.BatchNorm1d(hidden_size * 3)
            self.hid4 = nn.Linear(3 * hidden_size, 8 * hidden_size)
            self.dropout2 = nn.Dropout(0.1)
            self.hid5 = nn.Linear(8 * hidden_size, 16 * hidden_size)
            self.oupt = nn.Linear(16*hidden_size, output_length * prod(output_shape))
            # self.hid1 = nn.Linear(input_size, hidden_size * 6)  # 8-(10-10)-1
            # self.dropout1 = nn.Dropout(0.1)
            # self.bn1 = nn.BatchNorm1d(hidden_size * 6)
            # self.hid2 = nn.Linear(6 * hidden_size, hidden_size * 8)
            # self.bn2 = nn.BatchNorm1d(hidden_size * 8)
            # self.hid3 = nn.Linear(8 * hidden_size, hidden_size * 12)
            # self.bn3 = nn.BatchNorm1d(hidden_size * 12)
            # self.hid4 = nn.Linear(12 * hidden_size, 24*hidden_size)
            # self.dropout2 = nn.Dropout(0.1)
            # self.oupt = nn.Linear(24*hidden_size, output_length*prod(output_shape))

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.hid3.weight)
        nn.init.zeros_(self.hid3.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

    def forward(self, x):

        z = T.relu(self.hid1(x**2))
        # if x.shape[0] != 1:  # batch_size==1
        # z = self.bn1(z)
        # z = self.dropout1(z)
        z = T.relu(self.hid2(z))
        # z = self.bn2(z)
        # z = self.bn2(z)
        z = T.relu(self.hid3(z))
        # z = self.bn3(z)
        z = T.relu(self.hid4(z))
        z = T.relu(self.hid5(z))
        # z = self.dropout2(z)
        z = self.oupt(z)  # no activation

        return z
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_features, mid_features, out_features):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_features,mid_features)
        self.bn1 = nn.BatchNorm1d(mid_features)
        self.fc2 = nn.Linear(mid_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.shortcut = nn.Sequential()


    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks,input_size,hidden_size,output_length,output_shape):
        super(ResNet, self).__init__()
        # calculate the product of a list
        prod = lambda lst: reduce(lambda x, y: x * y, lst)
        self.fc_in= nn.Linear( input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer1 = self._make_layer(block, hidden_size, hidden_size, num_blocks[0])
        self.layer2 = self._make_layer(block, hidden_size, hidden_size, num_blocks[1])
        self.layer3 = self._make_layer(block, hidden_size, hidden_size, num_blocks[2])
        self.layer4 = self._make_layer(block, hidden_size, hidden_size, num_blocks[3])
        self.fc_out = nn.Linear(hidden_size, output_length*prod(output_shape))

    def _make_layer(self, block, in_features,out_features, num_blocks):

        layers = []
        layers.append(block(in_features,out_features,out_features))
        for stride in range(num_blocks-1):
            layers.append(block(out_features,out_features,out_features))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.fc_in(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc_out(out)
        return out

def ResNet18(input_size,hidden_size,output_length,output_shape):
    return ResNet(BasicBlock, [2, 2, 2, 2],input_size,hidden_size,output_length,output_shape)