import datetime
import cProfile,pstats,io
from matplotlib import pyplot as plt
import torch
import numpy as np
from ChannelMatrixEvaluation import test_configurations_capacity
from rate_model import capacity_loss
from collections import OrderedDict
import os
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
def print_train_iteration(start_time,batch_idx,epoch,max_epochs,train_ldr,pr,NMSE_lst):
    print("progress: "+str(100*(batch_idx/len(train_ldr))).split(".")[0]+"%"+" Train NMSE: {0:.4f}".format(np.mean(NMSE_lst)))
    time_elapsed = datetime.datetime.now() - start_time
    time_per_iteration = time_elapsed / (batch_idx + 1)
    completed = epoch/max_epochs +(batch_idx/len(train_ldr))/max_epochs
    time_left = time_per_iteration*len(train_ldr)*(max_epochs-epoch-1)+time_per_iteration * (len(train_ldr) - batch_idx - 1)
    print("completed {0:.2%} time left: {1}".format(completed, str(time_left).split(".")[0]))
    if pr is not None and batch_idx == 40:
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        pr.dump_stats("train_time_profile.prof")
        exit()
def plot_train_epoch(Y,oupt,model_output_capacity):
    plt.plot(torch.std(Y, 0).t().cpu().detach().numpy(), 'b--')
    plt.plot(torch.std(oupt, 0).t().cpu().detach().numpy(), "r")
    plt.legend(["ground truth variation", "output variation"])
    plt.show()
    plt.plot((torch.std(oupt, 0) / (torch.std(Y, 0))).t().cpu().detach().numpy(), "g--")
    plt.title("output.std/GT.std()")
    plt.show()
    if not model_output_capacity:
        plt.plot(abs(Y[0, :]).t().cpu().detach().numpy(), 'b--')
        plt.plot(abs(oupt[0, :]).t().cpu().detach().numpy(), 'r')
        plt.legend(["ground truth", "output"])
    plt.show()
def test_dnn_optimization(physfad,opt_inp_lst,tx_x,tx_y,device,noise):
    # while (loss.item() > -10000 and iters < num_of_iterations):
    physfad_model_capacity_lst = []
    for i, opt_inp in enumerate(opt_inp_lst):
        physfad_capacity, physfad_H = test_configurations_capacity(physfad,opt_inp,tx_x,tx_y,device=device,noise=noise)
        # physfad_capacity = torch.sum(torch.abs(physfad_H))
        print("iter {0} physfad_c capacity {1}".format(20 * i, physfad_capacity))
        physfad_model_capacity_lst.append(physfad_capacity)
    np_dnn_physfad_capacity = np.array([x.detach().numpy() for x in physfad_model_capacity_lst])
    return np_dnn_physfad_capacity
def plot_model_optimization(time_lst,physfad_model_capacity_lst,model_capacity_lst,
                            physfad_time_lst,physfad_capacity_lst,
                            zogd_time_lst,zogd_capacity,
                            random_time_lst,random_capacity,device):
    # while (loss.item() > -10000 and iters < num_of_iterations):
    plots_lst = []
    legend_lst = []
    if time_lst is not None:


        delta_time_list = [(t-time_lst[0])/ datetime.timedelta(seconds=1) for t in time_lst]
        delta_time_list[0] = delta_time_list[0] + 0.001
        best_capacity = 0
        number_of_consecutive_non_increases = 0
        best_iteration = 0
        for i,capacity in enumerate(physfad_model_capacity_lst):
            number_of_consecutive_non_increases = number_of_consecutive_non_increases + 1
            if capacity>best_capacity:
                best_capacity = capacity
                number_of_consecutive_non_increases = 0
            best_iteration = i
            if number_of_consecutive_non_increases > 40:
                best_iteration = i-40
                break
        p1, = plt.plot(delta_time_list[:best_iteration],model_capacity_lst[:best_iteration])
        p2, = plt.plot(delta_time_list[:best_iteration],physfad_model_capacity_lst[:best_iteration])
        p1.set_label("DNN model")
        p2.set_label("DNN model (tested on physfad_c)")
        plots_lst.append(p1)
        plots_lst.append(p2)
        legend_lst.append("DNN model")
        legend_lst.append("DNN model (tested on physfad_c)")
    if physfad_time_lst is not None:
        physfad_delta_time_list = [(t-physfad_time_lst[0])/ datetime.timedelta(seconds=1) for t in physfad_time_lst]
        physfad_delta_time_list[0] = physfad_delta_time_list[0]+0.001
        p3, = plt.plot(physfad_delta_time_list, physfad_capacity_lst)
        p3.set_label("Physfad optimization")
        plots_lst.append(p3)
        legend_lst.append("Physfad optimization")
    if zogd_time_lst is not None:
        zogd_delta_time_list = [(t - zogd_time_lst[0]) / datetime.timedelta(seconds=1) for t in zogd_time_lst]
        zogd_delta_time_list[0] = zogd_delta_time_list[0] + 0.001
        p4, = plt.plot(zogd_delta_time_list, zogd_capacity)
        p4.set_label("Zero Order Gradient Descent")
        plots_lst.append(p4)
        legend_lst.append("Zero Order Gradient Descent")
    if random_time_lst is not None:
        random_delta_time_list = [(t - random_time_lst[0]) / datetime.timedelta(seconds=1) for t in random_time_lst]
        random_delta_time_list[0] = random_delta_time_list[0] + 0.001
        random_capacity = [max(random_capacity[:i+1]) for i,x in enumerate(random_capacity)]
        p5, = plt.plot(random_delta_time_list, random_capacity)
        p5.set_label("Random Search")
        plots_lst.append(p5)
        legend_lst.append("Random Search")
    plt.legend(plots_lst,legend_lst)
    plt.xlabel("seconds[s]")
    plt.ylabel("Channel Capacity")
    plt.xscale('log')
    plt.title("optimization of channel capacity")

    plt.show()
def plus_1_cyclic(counter,max_value,first_loop=True):
    counter = counter+1
    if counter >= max_value:
        counter = 0
        first_loop = False
    return counter, first_loop
def open_virtual_batch(batch):
    (X, X_gradients, tx_x, tx_y, Y_capacity, Y) = batch  # (predictors, targets)
    X = X[0]
    X_gradients = X_gradients[0]
    Y_capacity = Y_capacity[0]
    Y = Y[0]
    return (X, X_gradients, tx_x, tx_y, Y_capacity, Y)

def test_model(test_ldr,model,physfad,output_size,output_shape,model_output_capacity,device):
    count = 0
    if test_ldr is not None:
        test_NMSE = 0
        for (test_batch_idx, test_batch) in enumerate(test_ldr):
            (X_test,X_gradients, tx_x, tx_y, Y_test_Capacity, Y_test) = open_virtual_batch(test_batch)
            test_output = model(tx_x,tx_y,X_test)
            if model_output_capacity:
                test_NMSE += NMSE(test_output, Y_test_Capacity)
            else:
                test_NMSE += NMSE(test_output, Y_test)
            count = count + 1
        test_NMSE = test_NMSE / count
        if model_output_capacity:
            model_capacity = test_output[0, :]

        else:
            model_capacity = capacity_loss(test_output[0, :].reshape(1,output_size,output_shape[0],output_shape[1]), torch.ones(output_size,device=device), 1)
            # model_capacity = -np.inf
        physfad_capacity, _ = test_configurations_capacity(physfad, X_test[0, :], tx_x, tx_y, device)
        # physfad_capacity = -np.inf
    else:
        test_NMSE = -np.inf
        model_capacity = -np.inf
        physfad_capacity = -np.inf
    return test_NMSE,model_capacity,physfad_capacity

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
def zo_estimate_gradient(func, x_unristricted, tx_x, tx_y, epsilon, m, device):
    N = x_unristricted.shape[-1]
    batch_size = x_unristricted.shape[0]
    batch_of_rand_vecs = generate_m_random_points_on_Nsphere(batch_size,m,N,device)
    f_x_plus_eps = torch.zeros((batch_size,m), device=device,dtype=torch.float64)
    f_x_minus_eps = torch.zeros((batch_size,m), device=device,dtype=torch.float64)
    with torch.no_grad():
        # TODO: broadcast this(currently broadcasting only the random_points)..
        #  I need to combine both the batches and the locations into the same dimension
        for i,rand_vectors in enumerate(batch_of_rand_vecs):
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
    # physfad_capacity, physfad_H = test_configurations_capacity(physfad,rate_model,estOptInp, noise=noise,list_out=True)
    # physfad_grad = -torch.autograd.grad(physfad_capacity, estOptInp,retain_graph=True)[0]# TODO NOTE TO SELF: THIS SHOULD BE FIXED PROBABLY NEED TO CALCULATE EACH GRADIENT SEPERATLY
    # return physfad_grad
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

        Y_opt_capacity_i, Y_opt_gt_i = test_configurations_capacity(physfad, estOptInp_i,tx_x_i,tx_y_i, device=device, list_out=True,noise=noise)
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
def shrinkage(x,gamma):
    larger_then_gamma_mask = x>=gamma
    smaller_then_minus_gamma_mask = x<=-gamma
    else_mask = torch.bitwise_not(torch.bitwise_or(larger_then_gamma_mask,smaller_then_minus_gamma_mask))
    # else_mask = [not (lrg or sml) for (lrg,sml) in zip(larger_then_gamma_mask, smaller_then_minus_gamma_mask)]

    x[larger_then_gamma_mask] = (x-gamma)[larger_then_gamma_mask]
    x[smaller_then_minus_gamma_mask] = (x+gamma)[smaller_then_minus_gamma_mask]
    x[else_mask] = torch.zeros_like(x)[else_mask]
    return x

def save_fig(fname,path):
    """
    The function saves the current figure to the specified file by using plt.savefig with the file path obtained by joining FIGURES_PATH and fname
    :param fname:  File name or path to save the figure.
    """
    plt.savefig(os.path.join(path, fname), bbox_inches='tight', format='pdf', transparent=True)

