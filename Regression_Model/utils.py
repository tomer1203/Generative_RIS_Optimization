
from matplotlib import pyplot as plt
import torch
import numpy as np
import ChannelMatrixEvaluation
from collections import OrderedDict
import os
from functools import wraps
import time

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
class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


def test_dnn_optimization(physfad,opt_inp_lst,tx_x,tx_y,device,noise):
    # while (loss.item() > -10000 and iters < num_of_iterations):
    physfad_model_capacity_lst = []
    for i, opt_inp in enumerate(opt_inp_lst):
        physfad_capacity, physfad_H = ChannelMatrixEvaluation.test_configurations_capacity(physfad,opt_inp,tx_x,tx_y,device=device,noise=noise)
        # physfad_capacity = torch.sum(torch.abs(physfad_H))
        print("iter {0} physfad_c capacity {1}".format(5 * i, physfad_capacity))
        physfad_model_capacity_lst.append(physfad_capacity)
    np_dnn_physfad_capacity = np.array([x.detach().numpy() for x in physfad_model_capacity_lst])
    return np_dnn_physfad_capacity


def open_virtual_batch(batch):
    (X, X_gradients, tx_x, tx_y, Y_capacity, Y) = batch  # (predictors, targets)
    X = X[0]
    X_gradients = X_gradients[0]
    Y_capacity = Y_capacity[0]
    Y = Y[0]
    return (X, X_gradients, tx_x, tx_y, Y_capacity, Y)


def NMSE(estimations,ground_truth):
    estimations = torch.abs(estimations)
    estimations_c = estimations.cpu().detach().numpy()
    ground_truth_c = ground_truth.cpu().detach().numpy()
    return 100 * (np.mean((estimations_c - ground_truth_c) ** 2) /
                  np.mean(ground_truth_c**2))

def generate_m_random_points_on_Nsphere(batch_size,m,N,device):
    random_mat = np.random.random((batch_size,m,N))*2 - 1
    norm_mat = np.expand_dims(np.linalg.norm(random_mat,axis=2),axis=2)
    tensor_output = torch.tensor(random_mat / norm_mat,device=device)
    return tensor_output
def zo_estimate_gradient(func, x_unristricted, tx_x, tx_y, epsilon, m, device,broadcast_tx):
    N = x_unristricted.shape[-1]
    batch_size = x_unristricted.shape[0]
    batch_of_rand_vecs = generate_m_random_points_on_Nsphere(batch_size,m,N,device)
    f_x_plus_eps = torch.zeros((batch_size,m), device=device,dtype=torch.float64)
    f_x_minus_eps = torch.zeros((batch_size,m), device=device,dtype=torch.float64)
    with torch.no_grad():
        # TODO: broadcast this(currently broadcasting only the random_points)..
        #  I need to combine both the batches and the locations into the same dimension
        for i,rand_vectors in enumerate(batch_of_rand_vecs):
            if broadcast_tx:
                current_tx_x, current_tx_y = tx_x, tx_y
            else:
                current_tx_x, current_tx_y = tx_x[i].unsqueeze(0), tx_y[i].unsqueeze(0)
            current_x = x_unristricted[i].type(torch.float64)
            # get sample of points
            normalized_x_plus_epsilon = current_x+epsilon*rand_vectors
            normalized_x_minus_epsilon =current_x-epsilon*rand_vectors
            # test function on sample
            f_x_plus_eps[i] = func(normalized_x_plus_epsilon,current_tx_x,current_tx_y)
            f_x_minus_eps[i] = func(normalized_x_minus_epsilon,current_tx_x,current_tx_y)
            # TODO: and then I need to reseperate the batches and points into their own dimensions..
    return torch.sum((f_x_plus_eps-f_x_minus_eps).unsqueeze(2)*batch_of_rand_vecs/(2*epsilon),dim=1)/m

def cosine_similarity(A,B):
    eps = 1e-10
    A, B = A+eps, B+eps
    norm = torch.sqrt((A*A).sum())*torch.sqrt((B*B).sum())
    dot_product = (A*B).sum()
    return dot_product/norm

def get_physfad_grads(estOptInp,tx_x,tx_y,physfad,device,noise=None,broadcast_tx=True):
    physfad_grad = torch.zeros(estOptInp.shape, device=physfad.device)

    for i in range(estOptInp.shape[0]): # for every element in batch
        # print("physfad " + str(i))
        estOptInp_i = estOptInp[i]
        if broadcast_tx:
            tx_x_i = tx_x
            tx_y_i = tx_y
        else:
            tx_x_i = tx_x[i].unsqueeze(0)
            tx_y_i = tx_y[i].unsqueeze(0)

        Y_opt_capacity_i, Y_opt_gt_i = ChannelMatrixEvaluation.test_configurations_capacity(physfad, estOptInp_i,tx_x_i,tx_y_i, device=device, list_out=True,noise=noise)
        physfad_grad[i] = torch.autograd.grad(-Y_opt_capacity_i, estOptInp_i, retain_graph=True)[0]
    return physfad_grad

def cosine_score(A,B):
    return 50*(cosine_similarity(A,B)+1)

def get_gradient_score(model_rate,estOptInp,tx_x,tx_y,physfad,device,noise=None):
    physfad_grad = get_physfad_grads(estOptInp,tx_x,tx_y,physfad,device=device,noise=noise)
    model_grad = torch.autograd.grad(model_rate, estOptInp,retain_graph=True)[0]
    return cosine_score(physfad_grad,model_grad),physfad_grad,model_grad

def directional_derivative_accuracy(estOptInp_before,estOptInp_after,physfad_grad,step_size):
    zero_order_grad_approximation = (estOptInp_after-estOptInp_before)/step_size
    derivative_direction = zero_order_grad_approximation/torch.sqrt((zero_order_grad_approximation**2).sum())
    physfad_directional_mag = torch.dot(physfad_grad.squeeze(),derivative_direction.squeeze())
    physfad_projection = physfad_directional_mag * derivative_direction
    return torch.mean((physfad_projection-zero_order_grad_approximation)**2)
def copy_with_gradients(x,device=None):
    if device is None:
        device = x.device
    return x.clone().to(device).detach().requires_grad_(True)
def copy_without_gradients(x,device=None):
    if device is None:
        device = x.device
    return x.clone().to(device).detach().requires_grad_(False)


def save_fig(fname,path):
    """
    The function saves the current figure to the specified file by using plt.savefig with the file path obtained by joining FIGURES_PATH and fname
    :param fname:  File name or path to save the figure.
    """
    plt.savefig(os.path.join(path, fname), bbox_inches='tight', format='pdf', transparent=True)

