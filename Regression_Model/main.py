import numpy as np
import torch
import torch
from matplotlib import pyplot as plt
import scipy.io
import time
import random
from models import *
from dataset import *
from ChannelMatrixEvaluation import (test_configurations_capacity,
                                       physfad_channel_optimization,
                                     zeroth_grad_optimization,random_search_optimization)
from PhysFadPy import physfad_c
from rate_model import capacity_loss
import datetime
from functools import reduce
import cProfile,pstats,io
from utils import (print_train_iteration, plot_train_epoch, plot_model_optimization, plus_1_cyclic, test_model,
                   NMSE,test_dnn_optimization, cosine_similarity,cosine_score,get_gradient_score,get_physfad_grads,
                   directional_derivative_accuracy,copy_with_gradients,copy_without_gradients,open_virtual_batch,
                   estimate_gradient,shrinkage,save_fig)
# -----------------------------------------------------------


MSE_Loss_torch1 = nn.MSELoss()
MSE_Loss_torch2 = nn.MSELoss()


# -----------------------------------------------------------
def My_MSE_Loss(X,Y,Y_capacity,batch_size,output_size,output_shape,model_output_capacity,calc_capacity,device,capacity_maximization_on=False):

    if model_output_capacity:
        Y=Y_capacity.unsqueeze(1)
    # else:
        # X_Norm = (torch.abs(X) / torch.norm(torch.abs(X), p=1, dim=1).unsqueeze(1))+epsilon
        # Y_Norm = (torch.abs(Y) / torch.norm(torch.abs(Y), p=1, dim=1).unsqueeze(1))+epsilon
        # KL_divergence = torch.sum(X_Norm * torch.log(X_Norm / Y_Norm))
        # mean_MSE = torch.sum((torch.norm(torch.abs(X), p=1, dim=1).unsqueeze(1) - torch.norm(torch.abs(Y), p=1,
        #                                                                                      dim=1).unsqueeze(1)) ** 2)

    if X.shape[0]==1:
        min_std = torch.Tensor(0)
    else:
        # min_std = MSE_Loss_torch2(torch.std(X,0),torch.std(Y,0))
        min_std = (torch.std(X, 0) - torch.std(Y, 0)) ** 2

    MSE_Loss = torch.mean((X-Y)**2)
    # MSE_Loss = MSE_Loss_torch1(X,Y)
    if calc_capacity and not model_output_capacity:
        X_capacity = capacity_loss(X.reshape(batch_size,output_size,output_shape[0],output_shape[1]),list_out=True,device=device)
        capacity_mse = (X_capacity-Y_capacity)**2
    # sum_MSE = (torch.sum(X)-torch.sum(Y))**2

    # print("MSE ", end="")
    # print(MSE_Loss)
    # print("kl ", end="")
    # print(100*KL_divergence)
    # print("mean ", end="")
    # print(mean_MSE / 100)
    # print("variance ", end="")
    # print(torch.sum(min_std)/5)


    # loss = KL_divergence
    loss = MSE_Loss
    # loss = loss + 30*KL_divergence
    loss = loss + 100*torch.mean(min_std)# 100
    # loss = loss + 5*min_std# 100
    if calc_capacity and not model_output_capacity:
        loss = loss + torch.mean(capacity_mse)/2
    if capacity_maximization_on and not model_output_capacity:
        loss = loss - torch.mean(X_capacity)/3
        # print("capacity ", end="")
        # print(torch.mean(capacity_mse)/10)
    # loss = loss + mean_MSE/100
    # loss = MSE_Loss+50*KL_divergence+100*torch.sum(min_std)#+mean_MSE/100
    # loss = 100*KL_divergence+mean_MSE/100#+120*torch.sum(capacity_mse)
    # if torch.any(~torch.isfinite(loss)):
    #     print("non finite value detected")
    return loss
def derivative_loss(deriv_gt,deriv_pred):
    return ((deriv_pred-deriv_gt)**2).mean()
def optimize_n_iterations(X,tx_x,tx_y,n,net,batch_size,output_size,output_shape,lr,cut_off=-np.inf,model_output_capacity=False,device="cpu"):
    loss = torch.Tensor([1])
    optimizer = torch.optim.Adam([X],lr=lr)#weight_decay = 0.00001# weight_decay = 0.001
    iters = 0
    net.eval()

    while (loss.item()>cut_off and iters < n):
        optimizer.zero_grad()
        pred = net(tx_x,tx_y,X)
        if model_output_capacity:
            loss = -torch.sum(pred)
            loss = loss/batch_size
        else:
            loss = -capacity_loss(pred.reshape(batch_size,output_size,output_shape[0],output_shape[1]),device=device)
        # loss_with_reg = loss+torch.sum(torch.abs(X))
        loss.backward()
        optimizer.step()
        # X = X + torch.sqrt(torch.tensor(2 * lr, device=device)) * torch.randn(X.shape, device=device)
        iters = iters + 1

    return X,-loss.item()

def train_and_optimize(model,
                       loss_func, optimizer, physfad,
                       train_ds, test_ldr,
                       max_epochs, epoch_cut_off, ep_log_interval,max_batches_per_epoch,
                       batch_size, output_size, output_shape,
                       model_output_capacity,
                       test_optim_chance,initial_optim_steps,step_increase, optim_lr,
                       NMSE_LST_SIZE,load_new_data,RIS_config_file=None, H_realiz_file=None, capacity_file=None,gradients_file=None,device='cpu'):
    NMSE_lst = np.zeros(NMSE_LST_SIZE)
    time_list_size = 200
    time_list_idx = 0
    first_time_loop = True
    iteration_time_list = [datetime.datetime.now()-datetime.datetime.now()]*time_list_size
    NMSE_idx = 0
    optim_step = initial_optim_steps
    optimized_inputs = None
    optimized_outputs = None
    optimized_capacity = None
    current_time = datetime.datetime.now()

    model.train()
    if load_new_data:
        train_ds = RISDataset(RIS_config_file, H_realiz_file, capacity_file,gradients_file,physfad=physfad,
                              batch_size=batch_size,output_size=output_size, output_shape=output_shape, device=device)

    train_ldr = T.utils.data.DataLoader(train_ds,batch_size=1, shuffle=True)
    for epoch in range(0, max_epochs):
        T.manual_seed(1 + epoch)

        print("epoch {0} train dataset length {1}".format(epoch,batch_size*len(train_ldr)))
        for (batch_idx, batch) in enumerate(train_ldr):
            last_time = current_time
            current_time = datetime.datetime.now()
            iteration_time_list[time_list_idx] = current_time-last_time

            (X,X_gradients,tx_x,tx_y, Y_capacity, Y) = open_virtual_batch(batch)  # (predictors, targets)
            for param in model.parameters():  # fancy zero_grad
                param.grad = None
            oupt = model(tx_x,tx_y,X)
            loss_val = loss_func(oupt, Y, Y_capacity, oupt.shape[0], output_size, output_shape,model_output_capacity,calc_capacity=False,device=device)  # avg per item in batch
            grad_list = []
            X = copy_with_gradients(X, device)
            # for i in range(X.shape[0]):
            #     # print("model " + str(i))
            #     X_i = X[i].unsqueeze(0)
            #     Y_model_i = model(tx_x,tx_y,X_i)
            #     grad_list.append(torch.autograd.grad(-Y_model_i, X_i, retain_graph=True, create_graph=True)[0])
            # model_grad = torch.vstack(grad_list)
            # grad_loss = cosine_similarity(X_gradients, model_grad)
            # if batch_idx % 20 == 0:
            #     print(cosine_score(X_gradients, model_grad))
            # total_loss = - 200*grad_loss
            total_loss = loss_val #- 200*grad_loss
            # total_loss = loss_val / 100 - 2000 * grad_loss
            total_loss.backward()  # compute gradients
            optimizer.step()  # update wts
            if model_output_capacity:
                NMSE_lst[NMSE_idx] = NMSE(oupt, Y_capacity.unsqueeze(1))
            else:
                NMSE_lst[NMSE_idx] = NMSE(oupt, Y)
            NMSE_idx,_ = plus_1_cyclic(NMSE_idx,NMSE_LST_SIZE)
            time_list_idx,first_time_loop = plus_1_cyclic(time_list_idx,time_list_size,first_time_loop)


            # print("progress: "+str(100*(batch_idx/len(train_ldr))).split(".")[0]+"%"+" Train NMSE: {0:.4f}".format(np.mean(NMSE_lst)))
            # if batch_idx/len(train_ldr) > epoch_cut_off:
            if batch_idx > max_batches_per_epoch-1:
                break

            if random.random() < test_optim_chance:

                # optimize for optim_steps steps
                a_t = datetime.datetime.now()
                X_copy = copy_with_gradients(X,device)
                number_of_optimization_steps = random.randint(1,optim_step)
                X_opt,model_capacity = optimize_n_iterations(X_copy,tx_x,tx_y,
                                                             number_of_optimization_steps,
                                                             model,
                                                             batch_size, output_size, output_shape,
                                                             lr=optim_lr,
                                                             model_output_capacity=model_output_capacity,
                                                             device=device)

                # NOTICE: remove the requires grad if not using the derivative loss feature
                # X_opt = X_opt.clone().detach().requires_grad_(True).to(device)
                X_opt = copy_with_gradients(X_opt,device)

                # get the accurate capacity and channel for the optimized inputs
                # NOTICE: re-add the "with torch.no_grad" if not using the derivative loss feature
                # with torch.no_grad():
                Y_opt_capacity,Y_opt_gt = test_configurations_capacity(physfad,X_opt,tx_x,tx_y,device=device,list_out=True)

                Y_opt_gt_flattended = Y_opt_gt.reshape(batch_size,-1)

                # calculate loss
                model.train()
                optimizer.zero_grad()
                Y_model_opt = model(tx_x,tx_y,X_opt)
                loss_val = loss_func(Y_model_opt, torch.abs(Y_opt_gt_flattended), Y_opt_capacity, Y_model_opt.shape[0],
                                     output_size,output_shape,
                                     model_output_capacity,calc_capacity=True,device=device,capacity_maximization_on=True) # avg per item in batch

                physfad_grads = get_physfad_grads(X_opt, tx_x, tx_y, physfad, device)
                # model_grad = torch.zeros(X_opt.shape,device=device,requires_grad=True)
                # grad_list = []
                # for i in range(batch_size):
                #     # print("model "+str(i))
                #     X_opt_i = X_opt[i].unsqueeze(0)
                #     Y_model_opt_i = model(tx_x,tx_y,X_opt_i)
                #     grad_list.append(torch.autograd.grad(-Y_model_opt_i,X_opt_i, retain_graph=True, create_graph=True)[0])
                # model_grad = torch.vstack(grad_list)
                #
                # grad_loss = cosine_similarity(physfad_grads,model_grad)
                # total_loss = loss_val/100+grad_loss*1000
                total_loss = loss_val
                # gradient_score = get_gradient_score(loss_val, X_opt, parameters, W, torch.tensor(1, device=device),
                #                                     device)
                # gradient_score = cosine_score(physfad_grads, model_grad)
                # print(gradient_score)
                # backpropegate
                total_loss.backward()
                optimizer.step()  # update wts

                if model_output_capacity:
                    optimized_NMSE = NMSE(Y_model_opt, Y_opt_capacity.unsqueeze(1))
                else:
                    optimized_NMSE = NMSE(Y_model_opt, torch.abs(Y_opt_gt_flattended))
                # time_elapsed = datetime.datetime.now() - start_time
                time_per_iteration = reduce(lambda x, y: x + y, iteration_time_list) / (time_list_idx if first_time_loop else time_list_size)
                epochs_left = max_epochs-epoch-1
                if test_optim_chance<0 or test_optim_chance>1:
                    geometric_series_factor = -np.inf
                else:
                    growth_rate = 1+epoch_cut_off*test_optim_chance
                    geometric_series_factor = ((1-(growth_rate)**epochs_left)/(1-growth_rate)) - 1
                length_of_current_epoch = len(train_ldr)*epoch_cut_off
                time_left = time_per_iteration * (length_of_current_epoch*geometric_series_factor + length_of_current_epoch - batch_idx - 1)
                time_left2 = time_per_iteration*(max_batches_per_epoch*epochs_left+(max_batches_per_epoch-batch_idx))
                # print(str(time_left2).split(".")[0]+"s",end = " ")
                print("E{0} I{1:.0f}% {2} steps: {3} train NMSE: {4:.4f} optimized train NMSE: {5:.4f} model capacity {6:.4f} physfad_c capacity {7:.4f} {8:.4f} sum(X): {9:.0f}  {10:.0f}".format(
                    str(epoch),
                    # 100 * batch_idx / (epoch_cut_off*len(train_ldr)),
                    100 * (batch_idx) / max_batches_per_epoch,
                    str(time_left2).split(".")[0]+"s",
                    number_of_optimization_steps,
                    np.mean(NMSE_lst),
                    optimized_NMSE,
                    model_capacity,
                    Y_capacity.mean().cpu().detach().numpy(),
                    Y_opt_capacity.mean().cpu().detach().numpy(),
                    torch.sum(X_opt).cpu().detach().numpy()/ batch_size,
                    torch.sum(X).cpu().detach().numpy()/ batch_size))

                # remove the grads and the requires_grad to ensure that we get only the values
                X_opt = copy_without_gradients(X_opt,device)
                X_opt_gradients = copy_with_gradients(physfad_grads,device)
                Y_opt_gt_flattended = copy_without_gradients(Y_opt_gt_flattended,device)
                Y_opt_capacity = copy_without_gradients(Y_opt_capacity,device)

                if torch.any(~torch.isfinite(X_opt)):
                    print("non finite value detected")
                # add optimized_input,Y_opt to next epochs dataset.
                if optimized_inputs is None:
                    # optimized_inputs = physfad.fill_ris_config(X_opt,device)
                    optimized_inputs = X_opt
                    optimized_gradients = X_opt_gradients
                    optimized_outputs = torch.abs(Y_opt_gt_flattended)
                    optimized_capacity = Y_opt_capacity
                else:
                    # optimized_inputs = torch.vstack([optimized_inputs,fill_ris_config(X_opt,device)])
                    optimized_inputs = torch.vstack([optimized_inputs,X_opt])
                    optimized_gradients = torch.vstack([optimized_gradients,X_opt_gradients])
                    optimized_outputs = torch.vstack([optimized_outputs,torch.abs(Y_opt_gt_flattended)])
                    optimized_capacity = torch.hstack([optimized_capacity,Y_opt_capacity])
        # add to dataset
        train_ds.add_new_items(optimized_inputs,optimized_gradients,optimized_outputs,optimized_capacity)
        optimized_inputs,optimized_outputs,optimized_capacity = None,None,None
        if train_ds.dataset_changed:
            train_ds.save_dataset(RIS_config_file,gradients_file, H_realiz_file, capacity_file)
            train_ldr = T.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
        optim_step = optim_step+step_increase

def annealed_langevin(model,physfad,starting_configuration,tx_x,tx_y, number_of_iterations = 5,tau=torch.tensor(0.01),device=torch.device("cpu")):
    model.eval()
    alpha_list = torch.linspace(1,0.01,10)
    batch_size = starting_configuration.shape[0]
    ris_configuration = starting_configuration
    rate_list = []
    for i,alpha in enumerate(alpha_list):
        for k in range(number_of_iterations):
            alpha_v = torch.ones([batch_size,1])

            normal_noise = torch.randn_like(ris_configuration)
            denoiser_output = model(torch.hstack([ris_configuration,tx_x,tx_y,torch.sqrt(alpha_v)]))
            # TODO: check the constant. It might be 1/alpha without the square
            score_function = (1/(alpha)) * (torch.sqrt(1-alpha)*denoiser_output - ris_configuration)
            ris_configuration = ris_configuration+tau*score_function+torch.sqrt(2*tau)*normal_noise
        if torch.any(~torch.isfinite(physfad(ris_configuration, tx_x, tx_y))):
            print("recognized non-finite value in physfad")
        rate = capacity_loss(physfad(ris_configuration, tx_x, tx_y), sigmaN=torch.tensor(1, dtype=torch.float64), device=device).item()
        rate_list.append((i*number_of_iterations, rate))
        config_norm = torch.norm(ris_configuration).item()
        print("alpha",alpha.item(),"k",k,"rate",rate,"norm",config_norm)
    model.train()
    return rate_list


def diffusion_training(model_forward,model_diffusion,train_ldr,test_ldr, optimizer,optimizer_diffusion, physfad, max_epochs, ep_log_interval, output_size, output_shape,
                       model_output_capacity, device="cpu"):

    # X=torch.randn([1,264],device=device)
    batch_size=8
    x_tx_orig = torch.tensor([0, 0, 0]).repeat(batch_size,1).to(device)
    y_tx_orig = torch.tensor([4, 4.5, 5]).repeat(batch_size,1).to(device)
    capacity_physfad = lambda x,tx_x,tx_y: -capacity_loss(physfad(x, tx_x, tx_y), sigmaN=torch.tensor(1,dtype=torch.float64), device=device)
    sigma_min=0.01
    sigma_max=1
    for i in range(1000):
        # (X, _, tx_x, tx_y, _, Y_ground_truth) = next(iter(train_ldr))
        # X = X[0, 0, :].unsqueeze(0).clone().detach().requires_grad_(True).to(device)
        X = torch.randn([batch_size, 135], device=device,dtype=torch.float64)
        tx_x_diff = 19.5 * torch.randn([batch_size, 3], device=device,dtype=torch.float64) - 3.3
        tx_y_diff = 11.5 * torch.randn([batch_size, 3], device=device,dtype=torch.float64) - 2.8
        tx_x, tx_y = x_tx_orig + tx_x_diff, y_tx_orig + tx_y_diff
        # tx_x, tx_y = x_tx_orig , y_tx_orig
        sigma = torch.FloatTensor(batch_size,1).uniform_(sigma_min, sigma_max).to(device).type(torch.float64)
        optimizer.zero_grad()
        optimizer_diffusion.zero_grad()
        improved_X = model_diffusion(torch.hstack([X,tx_x,tx_y,sigma]))
        # H_approx = model_forward(tx_x,tx_y,improved_X)
        epsilon = 0.0001
        x_distance = ((improved_X - X) ** 2).sum(dim=1)
        sigma_distance = ((x_distance - sigma.squeeze(1)) ** 2).mean()
        sigma_grad = torch.autograd.grad(-sigma_distance, improved_X, retain_graph=True, create_graph=True)[0]
        # acc_gradients = get_physfad_grads(improved_X, tx_x, tx_y, physfad, device, noise=None)
        # plt.plot(acc_gradients.squeeze(0))
        # M_list = [4,8,16,32,64,128]
        # epsilon_list = [0.01,0.001,0.0001,0.00001]
        # grad_dict = {}
        # legened_list = ["acc"]
        # for m in M_list:
        #     for eps in epsilon_list:
        #         grad_dict[(m,eps)] = estimate_gradient(capacity_physfad, improved_X, tx_x, tx_y, eps, m, device)
        #         if m == 16 and eps == 0.001:
        #             print("testing specific shrinkage")
        #         plt.plot(grad_dict[(m,eps)].squeeze())
        #         print(m,"-",eps,": ", cosine_score(grad_dict[(m,eps)], acc_gradients))
        #         legened_list.append(str(m)+"-"+str(eps))
        # plt.legend(legened_list)
        # plt.show()

        grad_inp_16 = estimate_gradient(capacity_physfad, improved_X, tx_x, tx_y, epsilon, 16, device)



        # grad_inp_8 = estimate_gradient(capacity_physfad, improved_X, tx_x, tx_y, epsilon, 8, device)

        grad_inp = grad_inp_16
        total_grad = grad_inp#+sigma_grad # TODO: reintroduce the sigma value here

        # rate_approx = capacity_loss(H_approx.reshape(H_approx.shape[0], output_size, output_shape[0], output_shape[1]), list_out=True,
        #               device=device)
        # rate,H = test_configurations_capacity(physfad,improved_X,tx_x,tx_y,device,list_out=False,noise=None)
        # H_mse = ((H.reshape([H_approx.shape[0],-1])-H_approx)**2).mean()
        # rate_mse = (rate-rate_approx)**2
        # minus_rate = -rate_approx / (1+rate_mse)
        # minus_rate = -rate
        # minus_H_mse = -H_mse
        # minus_rate.backward(retain_graph=True)
        # minus_H_mse.backward()
        improved_X.backward(grad_inp)
        # optimizer.step()
        optimizer_diffusion.step()
        # print(rate.item())
        if i % 16 == 0:
            acc_rate = 0
            # print(len(test_ldr))
            ald_capacity_avg = np.zeros(10)
            ald_iter_list = np.zeros(10)
            zogd_capacity_avg = np.zeros(50)
            physfad_capacity_avg = np.zeros(50)
            for batch in test_ldr:
                (X, X_gradients, tx_x, tx_y, Y_capacity, Y) = open_virtual_batch(batch)
                X = X.type(torch.float64)
                X = X[0].unsqueeze(0)
                improved_X = model_diffusion(torch.hstack([X,tx_x, tx_y,torch.tensor(1,device=device).unsqueeze(0).unsqueeze(0)]))
                ald_results = annealed_langevin(model_diffusion, physfad, X, tx_x, tx_y, device=device)
                # ald_iters = list(ald_iters)
                # ald_capac = list(ald_capac)
                for i,(iter,capac) in enumerate(ald_results):
                    ald_capacity_avg[i] += capac
                    ald_iter_list[i] = iter
                (zogd_time_lst, zogd_capacity, zogd_gradient_score) = zeroth_grad_optimization(device, physfad,
                                                                                               X.clone().detach().requires_grad_(
                                                                                                   False).to(
                                                                                                   device), tx_x,
                                                                                               # change require grad to false
                                                                                               tx_y,
                                                                                               noise_power=1,
                                                                                               num_of_iterations=2)
                for i,capac in enumerate(zogd_capacity):
                    zogd_capacity_avg[i] += capac
                # (physfad_time_lst, physfad_capacity, physfad_inputs) = (
                #     physfad_channel_optimization(device, physfad,copy_with_gradients(X,device),tx_x, tx_y,noise_power=1,
                #                                  learning_rate=0.1,num_of_iterations=2))  # 50
                # for i,capac in enumerate(physfad_capacity):
                #     physfad_capacity_avg[i] += capac

                # H_approx = model_forward(tx_x, tx_y,improved_X)
                # rate_approx = capacity_loss(H_approx.reshape(H_approx.shape[0], output_size, output_shape[0], output_shape[1]), list_out=True,
                #               device=device)
                rate, H = test_configurations_capacity(physfad, improved_X, tx_x, tx_y, device, list_out=False,
                                                       noise=None)
                acc_rate = acc_rate+rate
                # H_test_mse = ((abs(H).reshape([H_approx.shape[0],-1])-H_approx)**2).mean()/abs((H**2).mean())
            # print("test diffusion mean rate: ",(acc_rate/len(test_ldr)).item(),"test forward model accuracy(NMSE): ", H_test_mse.item())
            print("test diffusion mean rate: ",(acc_rate/len(test_ldr)).item())
            plt.plot(ald_iter_list,ald_capacity_avg/len(test_ldr))
            plt.plot(zogd_capacity_avg/len(test_ldr))
            # plt.plot(physfad_capacity_avg/len(test_ldr))
            plt.legend(["ald", "zogd"])
            plt.title("capacity at iteration for different algorithms") # average results over a small test set of 8 tx locations
            # plt.show()
            save_fig("diffusion_capacity_per_iteration.pdf","plots")
            plt.show()

        # print(rate.item())



def train(model, loss_func, optimizer,physfad, train_ldr, test_ldr, max_epochs, ep_log_interval, output_size, output_shape, model_output_capacity, NMSE_LST_SIZE, device):
    NMSE_lst = np.zeros(NMSE_LST_SIZE)
    variation_list = np.zeros(NMSE_LST_SIZE)
    capacity_list = np.zeros(NMSE_LST_SIZE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=3)
    NMSE_idx = 0
    NMSE_Train = []
    NMSE_TEST = []
    model.train()  # set mode
    # pr = cProfile.Profile()
    # pr.enable()
    for epoch in range(0, max_epochs):
        T.manual_seed(1 + epoch)  # recovery reproducibility
        epoch_loss = 0  # for one full epoch
        start_time = datetime.datetime.now()
        for (batch_idx, batch) in enumerate(train_ldr):
            (X,X_gradients,tx_x,tx_y,Y_capacity, Y) = open_virtual_batch(batch)  # (predictors, targets)

            # optimizer.zero_grad()  # prepare gradients
            for param in model.parameters(): # fancy zero_grad
                param.grad = None
            oupt = model(tx_x,tx_y,X)
            # print(torch.sum(oupt).item())
            loss_val = loss_func(oupt, Y,Y_capacity,oupt.shape[0], output_size,output_shape,model_output_capacity,calc_capacity=False,device=device)  # avg per item in batch
            # grad_list = []
            # X = copy_with_gradients(X,device)
            # for i in range(X.shape[0]):
            #     # print("model " + str(i))
            #     X_i = X[i].unsqueeze(0)
            #     Y_model_i = model(tx_x,tx_y,X_i)
            #     grad_list.append(torch.autograd.grad(-Y_model_i, X_i, retain_graph=True,create_graph=True)[0])
            # model_grad = torch.vstack(grad_list)
            # grad_loss = cosine_similarity(X_gradients, model_grad)
            # if batch_idx%20 == 0:
            #     print(cosine_score(X_gradients, model_grad))
            # total_loss = - 200*grad_loss
            total_loss = loss_val #- 200*grad_loss
            # total_loss = loss_val/100 - 2000*grad_loss
            epoch_loss += loss_val.item()  # accumulate avgs
            total_loss.backward()  # compute gradients
            optimizer.step()  # update wts
            if model_output_capacity:
                NMSE_lst[NMSE_idx] = NMSE(oupt, Y_capacity.unsqueeze(1))
                variation_list[NMSE_idx] = (torch.mean(torch.std(oupt, 0) / torch.std(Y_capacity, 0))).item()
            else:
                NMSE_lst[NMSE_idx] = NMSE(oupt, Y)
                variation_list[NMSE_idx] = (torch.mean(torch.std(oupt, 0) / torch.std(Y, 0))).item()
                estimated_capacity = capacity_loss(oupt.reshape(oupt.shape[0],output_size,output_shape[0],output_shape[1]),list_out=True,device=device)
                capacity_list[NMSE_idx] = NMSE(estimated_capacity,Y_capacity)
            ## DEBUG
            # physfad_capacity, physfad_H = test_configurations_capacity(parameters, fill_ris_config(X[0,:], device),
            #                                                            device)
            # optimization_nmse = NMSE(oupt[0,:], torch.abs(physfad_H).reshape(1, -1))
            # print("Matlab: {0}".format(NMSE_lst[NMSE_idx]))
            # print("Python: {0}\n\n".format(optimization_nmse))
            ## END DEBUG


            if False and batch_idx%300==0:
                print_train_iteration(start_time, batch_idx, epoch, max_epochs, train_ldr, pr=None,NMSE_lst=NMSE_lst)

            NMSE_idx,_ = plus_1_cyclic(NMSE_idx, NMSE_LST_SIZE)


        if epoch % ep_log_interval == 0:
            if epoch % 3*ep_log_interval ==0:
                if not model_output_capacity:
                    plot_train_epoch(Y,oupt,model_output_capacity)
            test_NMSE,model_capacity,physfad_capacity = test_model(None,model,physfad,output_size,output_shape,model_output_capacity,device)
            # test_NMSE,model_capacity,physfad_capacity = test_model(test_ldr,model,physfad,rate_model,output_size,output_shape,model_output_capacity,device)
            if test_NMSE is not -np.inf:
                scheduler.step(test_NMSE)
            else:
                scheduler.step(np.mean(NMSE_lst))
            # print("NMSE for test is "+str(test_NMSE))
            time_elapsed = datetime.datetime.now() - start_time
            time_per_epoch = time_elapsed / (epoch + 1)
            completed = epoch / max_epochs
            time_left = time_per_epoch  * (max_epochs - epoch - 1)
            print("epoch =%4d time left =%s  loss =%0.4f NMSE =%0.5f test_NMSE =%0.5f output normalized variation=%0.5f, training capacity NMSE=%0.5f model capacity %0.3f, phsyfad capacity %0.3f" % \
                  (epoch,str(time_left).split(".")[0], epoch_loss, np.mean(NMSE_lst), test_NMSE, np.mean(variation_list), np.mean(capacity_list), model_capacity,physfad_capacity))
            NMSE_Train.append(np.mean(NMSE_lst))
            NMSE_TEST.append(test_NMSE)
            # save checkpoint
            # dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            # fn = ".\\Log\\" + str(dt) + str("-") + \
            #      str(epoch) + "_checkpoint.pt"

            # info_dict = {
            #     'epoch': epoch,
            #     'net_state': net.state_dict(),
            #     'optimizer_state': optimizer.state_dict()
            # }
            # T.save(info_dict, fn)
    return (NMSE_Train, NMSE_TEST)
def generate_dataset(dataset_name,dataset_post_name,dataset_path,batch_size,dataset_size,physfad,input_size):
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    x_tx_orig = torch.tensor([0, 0, 0]).repeat(int(dataset_size/batch_size), 1).to(device)
    y_tx_orig = torch.tensor([4, 4.5, 5]).repeat(int(dataset_size/batch_size), 1).to(device)
    capacity_physfad = lambda x, tx_x, tx_y: -capacity_loss(physfad(x, tx_x, tx_y),
                                                            sigmaN=torch.tensor(1, dtype=torch.float64), device=device,list_out=True)


    X = torch.randn([dataset_size, input_size], device=device, dtype=torch.float64)
    tx_x_diff = 19.5 * torch.randn([int(dataset_size/batch_size), 3], device=device, dtype=torch.float64) - 3.3
    tx_y_diff = 11.5 * torch.randn([int(dataset_size/batch_size), 3], device=device, dtype=torch.float64) - 2.8
    tx_x, tx_y = x_tx_orig + tx_x_diff, y_tx_orig + tx_y_diff
    rate, H = test_configurations_capacity(physfad, X[0:batch_size], tx_x[0].unsqueeze(0), tx_y[0].unsqueeze(0), device, list_out=False,
                                           noise=None)
    rate_dataset = torch.zeros(dataset_size)
    H_dataset = torch.zeros([dataset_size,*H.shape[1:]],dtype=torch.complex64)

    for batch_idx in range(int(dataset_size/batch_size)):
        print("generating batches",dataset_post_name,": ", 100*batch_idx*batch_size/dataset_size,"%")
        rate, H = test_configurations_capacity(physfad, X[batch_size*batch_idx:batch_size*(batch_idx+1)], tx_x[batch_idx].unsqueeze(0), tx_y[batch_idx].unsqueeze(0), device, list_out=True, noise=None)
        rate_dataset[batch_size*batch_idx:batch_size*(batch_idx+1)] = rate
        H_dataset[batch_size*batch_idx:batch_size*(batch_idx+1)] = H

    ris_configuration_file_name = dataset_path+dataset_name+"_RISConfiguration"+dataset_post_name+".mat"
    H_mat_file_name = dataset_path+dataset_name+"_H_realizations"+dataset_post_name+".mat"
    rate_file_name = dataset_path+dataset_name+"_H_capacity"+dataset_post_name+".mat"
    tx_x_file_name = dataset_path+dataset_name+"_transmitter_x_location"+dataset_post_name+".mat"
    tx_y_file_name = dataset_path+dataset_name+"_transmitter_y_location"+dataset_post_name+".mat"

    scipy.io.savemat(ris_configuration_file_name,{"RISConfiguration":X.detach().numpy()})
    scipy.io.savemat(H_mat_file_name,{"sampled_Hs":H_dataset.detach().numpy()})
    scipy.io.savemat(rate_file_name,{"rate":rate_dataset.detach().numpy()})
    scipy.io.savemat(tx_x_file_name,{"x_tx_modified":tx_x.detach().numpy()})
    scipy.io.savemat(tx_y_file_name,{"y_tx_modified":tx_y.detach().numpy()})

def load_data(batch_size,output_size,output_shape,physfad,device):
    train_RIS_file = "..\\Data\\conditional_RISConfiguration.mat"#"..\\Data\\full_range_RISConfiguration.mat" # full_range_RISConfiguration
    # train_RIS_file = "..\\Data\\new_full_RISConfiguration.mat"
    train_H_file = "..\\Data\\conditional_H_realizations.mat"#"..\\Data\\full_range_H_realizations.mat" # full_range_H_realizations
    # train_H_file = "..\\Data\\new_full_H_realizations.mat"
    train_tx_file = "..\\Data\\"
    train_H_capacity_file = "..\\Data\\full_H_capacity.txt" # full_H_capacity
    train_ris_gradients_file = "..\\Data\\full_ris_gradients.pt"
    train_ds = RISDataset(train_RIS_file, train_H_file, train_H_capacity_file,train_ris_gradients_file,train_tx_file, calculate_capacity=True,calculate_gradients=True,physfad=physfad,only_fres=False,
                        batch_size=batch_size,virtual_batch_size=16,output_size=output_size, output_shape=output_shape, device=device)
    test_RIS_file = "..\\Data\\conditional_RISConfiguration_test.mat"
    test_H_file = "..\\Data\\conditional_H_realizations_test.mat"
    test_H_capacity_file = "..\\Data\\Test_full_H_capacity.txt"
    test_ris_gradients_file = "..\\Data\\Test_full_ris_gradients.pt"

    test_ds = RISDataset(test_RIS_file, test_H_file, test_H_capacity_file,test_ris_gradients_file,train_tx_file, calculate_capacity=True,calculate_gradients=True,physfad=physfad,only_fres=False,
                         batch_size=batch_size,virtual_batch_size=16,output_size=output_size, output_shape=output_shape, device=device)

    train_ldr = T.utils.data.DataLoader(train_ds,
                                        batch_size=1, shuffle=True)
    test_ldr = T.utils.data.DataLoader(test_ds,
                                       batch_size=1, shuffle=True)
    return train_ds,test_ds,train_ldr,test_ldr


def optimize(physfad, starting_inp,tx_x,tx_y,net,inp_size,output_size,optim_lr = 0.005, model_output_capacity=False,
             calaculate_physfad=True,device='cpu',num_of_iterations=1500,noise=1):
    net.train()
    # run gradiant descent to find best input configuration
    batch_size = 1
    # (X, _, _) = next(iter(train_ldr))  # (predictors, targets)
    # estOptInp = torch.randn([batch_size, inp_size], device=device)
    # estOptInp[estOptInp >= 0] = 5 - 2 * (estOptInp[estOptInp >= 0])
    # estOptInp[estOptInp < 0] = 1 + (0.15 + estOptInp[estOptInp < 0]) * 0.3
    # estOptInp = starting_inp[0, :].unsqueeze(0).clone().detach().requires_grad_(True).to(device)  # Activate gradients
    estOptInp = starting_inp
    time_lst = []
    opt_inp_lst = []
    model_capacity_lst = []
    gradient_score_acc_lst = []
    # estOptInp = torch.tensor([batch_size,inp_size],requires_grad=True,device=device).uniform_(0.9,5)
    # estOptInp = torch.from_numpy(np.random.uniform(low=0.9, high=5, size=(batch_size, inp_size)),device=device,requires_grad=True)
    # print(estOptInp)
    # print(X)
    Inp_optimizer = torch.optim.Adam([estOptInp], lr=optim_lr, weight_decay = 0.00001)


    loss = torch.Tensor([1])
    # num_of_iterations = 1500
    loss_list_size = 1
    iters = -loss_list_size
    loss_list = np.zeros(loss_list_size)
    # while (loss.item() > -10000 and iters < num_of_iterations):
    net.eval()
    pred = net(tx_x,tx_y,estOptInp)


    if model_output_capacity:
        loss = -torch.sum(pred)
        loss = loss / batch_size
    else:
        loss = -capacity_loss(pred.reshape(batch_size, output_size, 4, 3), sigmaN=noise,device=device)
        # loss = -torch.sum(pred)

    time_lst.append(datetime.datetime.now())
    opt_inp_lst.append(estOptInp.clone())
    model_capacity_lst.append(-loss.item())
    old_inputs = estOptInp.clone().detach().requires_grad_(False).to(device)

    while (iters < num_of_iterations):
        Inp_optimizer.zero_grad()
        pred = net(tx_x,tx_y,estOptInp)
        if model_output_capacity:
            loss = -torch.sum(pred)
            loss = loss/batch_size
        else:
            loss = -capacity_loss(pred.reshape(batch_size, output_size, 4, 3), sigmaN=noise,device=device)
            # loss = -torch.sum(pred)
        # loss_with_reg = loss+torch.sum(torch.abs(estOptInp))/1000
        # loss = -torch.sum(pred)

        # gradient_score,physfad_grad,model_grad = get_gradient_score(-loss,estOptInp,tx_x,tx_y,physfad,device=device)
        #
        # print(gradient_score)
        # gradient_score_acc_lst.append(gradient_score)
        # physfad_gradients = get_physfad_grads(estOptInp, parameters, W, noise, device)

        optim_lr_torch = torch.tensor(optim_lr, device=device)
        # directional_derivative_accuracy(old_inputs, estOptInp, physfad_grad, optim_lr_torch)
        # old_inputs = estOptInp.clone().detach().requires_grad_(False).to(device)

        loss.backward()
        Inp_optimizer.step()
        # estOptInp = estOptInp + torch.sqrt(2*optim_lr_torch) * torch.randn(estOptInp.shape,device=device)
        loss_list[iters % loss_list_size] = -loss.item()
        if iters % 20 == 0:
            if calaculate_physfad:
                physfad_capacity, physfad_H = test_configurations_capacity(physfad, estOptInp,tx_x,tx_y,device=device,noise=noise)

                if model_output_capacity:
                    optimization_nmse = -np.inf
                else:
                    optimization_nmse = NMSE(pred.reshape(120, 4, 3), torch.abs(physfad_H))
            else:
                physfad_capacity = -np.inf
                optimization_nmse = -np.inf
            print(
                "DNN: iter #{0} input distance from zero: {1:.2f} model capacity loss: {2:.4f} physfad_c capacity {3:.4f}, NMSE {4:.4f}".format(
                    iters, torch.abs(
                        estOptInp).sum().cpu().item() / batch_size, loss_list.mean() / batch_size, physfad_capacity,
                    optimization_nmse))

            time_lst.append(datetime.datetime.now())
            opt_inp_lst.append(estOptInp.clone())
            model_capacity_lst.append(-loss.item())
            # print(
            #     "iter #{0} input distance from zero: {1:.2f} model capacity loss: {2:.4f} physfad_c capacity {3:.4f}".format(
            #         iters, torch.abs(estOptInp).sum().cpu().item() / batch_size, loss_list.mean() / batch_size, physfad_capacity))
            # plt.plot(torch.mean(test_pred,dim=0).t().cpu().detach().numpy())
            # plt.legend(["asa,asd"])
            # plt.show()
        iters = iters + 1
    print("Done optimization")
    print(estOptInp)
    np.savetxt("optimal_parameters.txt", estOptInp[0, :].cpu().detach().numpy())
    return (time_lst,opt_inp_lst,model_capacity_lst,gradient_score_acc_lst)

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # force choosing cpu
    print(device)
    # 0. get started
    T.manual_seed(4)  # representative results
    np.random.seed(4)

    batch_size = 16# 320
    output_size = 120
    output_shape = (4, 3)
    inp_size = 135 # 264
    hidden_size =40
    model_output_capacity = False

    # 1. create DataLoader objects

        # 2. create network
    # net_forward = Net(inp_size,50,120,(4,3),False).to(device)
    net_diffusion = Net_diffusion(135+6+1,inp_size,50).to(device)
    net_forward = hypernetwork_and_main_model(inp_size,hidden_size,120,(4,3),model_output_capacity).to(device)

    # 3. train model
    max_epochs = 150
    ep_log_interval = 1
    lrn_rate = 0.00005*10
    optim_lr = 0.002 #0.008
    num_of_opt_iter = 3000
    load_model = False
    diffusion_mode = True
    training_mode = False
    activate_train_and_optimize = False  # can be added to the train mode
    find_optimal_lr = False
    optimize_model = True
    run_snr_graph = False
    loss_func = My_MSE_Loss
    # optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
    # optimizer = T.optim.Adam(net.hyp_net.parameters(), lr=lrn_rate)  # weight_decay=0.001
    optimizer_diffusion = T.optim.Adam(net_diffusion.parameters(), lr=lrn_rate)  # weight_decay=0.001
    optimizer = T.optim.Adam(net_forward.parameters(),lr=lrn_rate)
    physfad = physfad_c(device=device)
    physfad.set_configuration()
    print("\nbatch_size = %3d " % batch_size)
    print("loss = " + str(loss_func))
    print("optimizer = Adam")
    print("max_epochs = %3d " % max_epochs)
    print("train learning rate = %0.4f " % lrn_rate)
    print("optimization learning rate = %f " % optim_lr)


    NMSE_LST_SIZE = 10
    print("Collecting RIS configuration ")
    # generate_dataset("conditional","","..\\Data\\",16,128,physfad,input_size=135)
    generate_dataset("conditional","_test","..\\Data\\",16,64,physfad,input_size=135)
    train_ds,test_ds,train_ldr, test_ldr = load_data(batch_size,output_size,output_shape,physfad,device)
    if load_model:
        print("Loading model")
        # net_forward.load_state_dict(torch.load(".\\Models\\large_model_long_tr_op_loop.pt"))
        net_forward.load_state_dict(torch.load(".\\Models\\Full_Main_model.pt"))
        # net.load_state_dict(torch.load(".\\Models\\rate_model.pt"))
        # optimizer = T.optim.Adam([net_forward.hyp_net.parameters(),net_forward.main_net.linear_layers.parameters()], lr=lrn_rate)
        optimizer = T.optim.Adam(net_forward.parameters(), lr=lrn_rate)
        optimizer_diffusion = T.optim.Adam(net_diffusion.parameters(), lr=lrn_rate)
    if diffusion_mode:
        diffusion_training(net_forward,
                           net_diffusion,
                           train_ldr,
                           test_ldr,
                           optimizer,
                           optimizer_diffusion,
                           physfad,
                           max_epochs,
                           ep_log_interval,
                           output_size,
                           output_shape,
                           model_output_capacity,
                           device=device)
    if training_mode:
        print("\nStarting training with saved checkpoints")
        (NMSE_Train, NMSE_TEST) = train(net,
                                        loss_func,
                                        optimizer,
                                        physfad,
                                        train_ldr,
                                        test_ldr,
                                        max_epochs,
                                        ep_log_interval,
                                        output_size,
                                        output_shape,
                                        model_output_capacity,
                                        NMSE_LST_SIZE,
                                        device=device)
        print("Done Training")
        plt.plot(range(0,max_epochs,ep_log_interval),NMSE_Train, "r")
        plt.plot(range(0,max_epochs,ep_log_interval),NMSE_TEST, "b")
        plt.legend(["Train NMSE", "Test NMSE"])
        plt.xlabel("epochs")
        plt.show()
        torch.save(net.state_dict(),".\\Models\\Full_Main_model.pt")

    if device != torch.device('cpu'):
        print("Moving data to cpu")
        device = torch.device('cpu')
        net = net.to(device)
        physfad = physfad_c(device=device)
        physfad.set_configuration()
        train_ds, test_ds, train_ldr, test_ldr = load_data(batch_size, output_size, output_shape, physfad, device)
        optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)


    if activate_train_and_optimize:
        print("Starting deep-training loop with optimization")
        batch_size = 32
        max_epochs = 120
        initial_optim_steps = 10
        step_increase = 10
        test_optim_chance = 0.1 # 0.015
        epoch_cut_off = 0.05
        max_batches_per_epoch = 300 # 700
        load_new_data = False

        print("batch_size: " + str(batch_size))
        print("max_epochs: " + str(max_epochs))
        print("initial_optim_steps: " + str(initial_optim_steps))
        print("step_increase: " + str(step_increase))
        print("test_optim_chance: " + str(test_optim_chance))
        print("epoch_cut_off: " + str(epoch_cut_off))
        print("load_new_data: " + str(load_new_data))


        optimized_train_RIS_file = "..\\Data\\optimized_RISConfiguration.mat"
        optimized_train_H_file = "..\\Data\\optimized_H_realizations.mat"
        optimized_train_H_capacity_file = "..\\Data\\optimized_H_capacity.txt"
        optimized_gradients_file = "..\\Data\\optimized_channel_gradients.pt"
        train_and_optimize(net,
                           loss_func,
                           optimizer, physfad,
                           train_ds,test_ldr,
                           max_epochs, epoch_cut_off, ep_log_interval,max_batches_per_epoch,
                           batch_size, output_size, output_shape,
                           model_output_capacity,
                           test_optim_chance, initial_optim_steps, step_increase,optim_lr,
                           NMSE_LST_SIZE,
                           load_new_data, optimized_train_RIS_file, optimized_train_H_file, optimized_train_H_capacity_file,optimized_gradients_file,device)

        torch.save(net.state_dict(), ".\\Models\\Full_Main_model.pt")
    if find_optimal_lr:
        device_cpu = torch.device('cpu')

        print("Finding Best Learning Rate")

        (X, _, tx_x, tx_y, _, _) = next(iter(train_ldr))  # (predictors, targets)
        # (X,_,tx_x,tx_y, _, _) = next(iter(train_ldr))  # (predictors, targets)

        initial_inp = X[0, 0, :].unsqueeze(0).clone().detach().requires_grad_(True).to(device_cpu)
        learning_rates = np.arange(0.005,0.05,0.005)
        num_of_iter = 50
        num_of_lr = len(learning_rates)
        capacities_curves =np.zeros([num_of_lr,num_of_iter])
        for i,lr in enumerate(learning_rates):
            (physfad_time_lst, physfad_capacity, physfad_inputs) = physfad_channel_optimization(device_cpu, physfad,
                                                                                            copy_with_gradients(
                                                                                                initial_inp,
                                                                                                device_cpu), tx_x, tx_y,
                                                                                            noise_power=1,learning_rate=lr,
                                                                                            num_of_iterations=num_of_iter)  # 50
            capacities_curves[i] = physfad_capacity
        plt.plot(capacities_curves.T)
        plt.legend([str(x) for x in learning_rates])
        plt.show()

    if optimize_model:
        device_cpu = torch.device('cpu')
        device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # optimized_train_RIS_file = "..\\Data\\large_model_optimized_RISConfiguration.mat"
        # optimized_train_H_file = "..\\Data\\large_model_optimized_H_realizations.mat"
        # optimized_train_H_capacity_file = "..\\Data\\large_model_optimized_H_capacity.txt"
        # train_ds = RISDataset(optimized_train_RIS_file, optimized_train_H_file, optimized_train_H_capacity_file, calculate_capacity=False,
        #                       output_size=output_size, output_shape=output_shape, device=device)
        #
        # train_ldr = T.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        (X,_,tx_x,tx_y, _, _) = next(iter(train_ldr))  # (predictors, targets)
        initial_inp = X[0,0, :].unsqueeze(0).clone().detach().requires_grad_(True).to(device)
        print("Running Optimization on DL Model")
        physfad_time_lst, physfad_capacity = None,None
        zogd_time_lst, zogd_capacity = None,None
        # (physfad_time_lst,physfad_capacity,physfad_inputs) = physfad_channel_optimization(device,initial_inp.clone().detach().requires_grad_(True).to(device))
        # (zogd_time_lst,zogd_capacity)                      = zeroth_grad_optimization(device,initial_inp.clone().detach().to(device))
        # (time_lst,opt_inp_lst,model_capacity_lst)          = optimize(initial_inp.clone().detach().requires_grad_(True),net,inp_size,output_size,optim_lr,model_output_capacity,num_of_iterations=num_of_opt_iter,calaculate_physfad=False,device=device)
        (X,_,tx_x,tx_y, _, Y_ground_truth) = next(iter(train_ldr))  # (predictors, targets)

        initial_inp = X[0,0, :].unsqueeze(0).clone().detach().requires_grad_(True).to(device_cpu)
        # H = physfad(copy_with_gradients(initial_inp,device_cpu), tx_x, tx_y)
        np.savetxt("Physfad_optimal_parameters.txt", initial_inp.cpu().detach().numpy())
        np.savetxt("Physfad_optimal_parameters.txt", initial_inp.cpu().detach().numpy())
        (zogd_time_lst, zogd_capacity, zogd_gradient_score) = zeroth_grad_optimization(device_cpu, physfad,
                                                                                       initial_inp.clone().detach().requires_grad_(
                                                                                           False).to(device_cpu), tx_x, # change require grad to false
                                                                                       tx_y,
                                                                                       noise_power=1,
                                                                                       num_of_iterations=50)
        (physfad_time_lst, physfad_capacity, physfad_inputs) = physfad_channel_optimization(device_cpu,physfad,
                                                                                            copy_with_gradients(initial_inp,device_cpu),tx_x,tx_y,
                                                                                            noise_power=1,learning_rate=0.1,
                                                                                            num_of_iterations=50)#50
        print(physfad_inputs)

        (rand_search_time_lst, random_search_capacity) = random_search_optimization(physfad,50, device_cpu, noise_power=1,
                                                                                    initial_inp=initial_inp.clone().detach().to(device_cpu),tx_x=tx_x,tx_y=tx_y)
        net = net.to(device_cuda)
        cuda_physfad = physfad_c(device=device_cuda)
        cuda_physfad.set_configuration()

        (time_lst, opt_inp_lst, model_capacity_lst,gradient_score_lst) = optimize(cuda_physfad,
            copy_with_gradients(initial_inp,device_cuda), copy_with_gradients(tx_x,device_cuda), copy_with_gradients(tx_y,device_cuda), net, inp_size, output_size, optim_lr,
            model_output_capacity, num_of_iterations=2000, noise=1,#num_of_opt_iter=2000?
            calaculate_physfad=False, device=device_cuda)

        dnn_physfad_capacity_lst = test_dnn_optimization(physfad,[x.cpu() for x in opt_inp_lst],tx_x,tx_y, device_cpu, noise=1)

        max_index = dnn_physfad_capacity_lst.argmax()

        time_lst = np.array(time_lst)
        physfad_time_lst = np.array(physfad_time_lst)
        zogd_time_lst = np.array(zogd_time_lst)
        rand_search_time_lst = np.array(rand_search_time_lst)

        physfad_capacity = np.array(physfad_capacity)
        zogd_capacity = np.array([x.detach().numpy() for x in zogd_capacity])
        # random_search_capacity = np.array([x.detach().numpy() for x in random_search_capacity])
        np.save(".\\outputs\\time_lst.npy",time_lst)
        np.save(".\\outputs\\physfad_time_lst.npy",physfad_time_lst)
        np.save(".\\outputs\\zogd_time_lst.npy",zogd_time_lst)
        np.save(".\\outputs\\rand_search_time_lst.npy",rand_search_time_lst)
        np.save(".\\outputs\\dnn_physfad_capacity_lst.npy",dnn_physfad_capacity_lst)
        np.save(".\\outputs\\physfad_capacity.npy",physfad_capacity)
        np.save(".\\outputs\\zogd_capacity.npy",zogd_capacity)
        np.save(".\\outputs\\random_search_capacity.npy",random_search_capacity)

        plot_model_optimization(time_lst,dnn_physfad_capacity_lst,model_capacity_lst,physfad_time_lst,physfad_capacity,zogd_time_lst,zogd_capacity,rand_search_time_lst,random_search_capacity,device)


    if run_snr_graph:
        device_cpu = torch.device('cpu')
        device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_snr_points = 15
        attempts = 6
        snr_values = np.linspace(0.5,60,num_snr_points)

        noise_array = 1/snr_values
        SNR_results = np.zeros([num_snr_points,7,attempts])
        for i,noise in enumerate(noise_array):
            for attempt in range(attempts):
                print(noise,i,attempt)
                (X, _, _) = next(iter(train_ldr))  # (predictors, targets)
                initial_inp = X[0, :].unsqueeze(0).clone().detach().requires_grad_(True).to(device_cpu)

                (physfad_time_lst, physfad_capacity, physfad_inputs) = physfad_channel_optimization(device_cpu,
                                                                                                    initial_inp.clone().detach().requires_grad_(True).to(device_cpu),
                                                                                                    noise_power=noise,num_of_iterations=150)
                (zogd_time_lst, zogd_capacity) = zeroth_grad_optimization(device_cpu, initial_inp.clone().detach().to(device_cpu),
                                                                          noise_power=noise,num_of_iterations=200)
                (rand_search_time_lst, random_search_capacity) = random_search_optimization(300,device_cpu,noise_power=noise)
                net = net.to(device_cuda)
                (time_lst, opt_inp_lst, model_capacity_lst) = optimize(physfad, initial_inp.to(device_cuda).clone().detach().requires_grad_(True), net, inp_size, output_size,optim_lr,
                    model_output_capacity,num_of_iterations=num_of_opt_iter,noise=noise,#num_of_opt_iter
                    calaculate_physfad=False, device=device_cuda)

                dnn_physfad_capacity_lst=test_dnn_optimization([x.cpu() for x in opt_inp_lst],device_cpu,noise=noise)

                max_index = dnn_physfad_capacity_lst.argmax()


                time_lst                = np.array(time_lst)
                physfad_time_lst        = np.array(physfad_time_lst)
                zogd_time_lst           = np.array(zogd_time_lst)
                rand_search_time_lst    = np.array(rand_search_time_lst)

                physfad_capacity = np.array(physfad_capacity)
                zogd_capacity = np.array([x.detach().numpy() for x in zogd_capacity])
                random_search_capacity = np.array([x.detach().numpy() for x in random_search_capacity])
                SNR_results[i, 0, attempt] = physfad_capacity[-1]
                SNR_results[i, 1, attempt] = zogd_capacity[-1]
                SNR_results[i, 2, attempt] = max(random_search_capacity)
                SNR_results[i, 3, attempt] = dnn_physfad_capacity_lst[max_index]
                SNR_results[i, 4, attempt] = physfad_capacity[physfad_time_lst-physfad_time_lst[0]< time_lst[max_index]-time_lst[0]][-1]
                SNR_results[i, 5, attempt] = zogd_capacity[zogd_time_lst-zogd_time_lst[0]< time_lst[max_index]-time_lst[0]][-1]
                SNR_results[i, 6, attempt] = max(random_search_capacity[rand_search_time_lst - rand_search_time_lst[0] < time_lst[max_index] - time_lst[0]])
        np.save(".\\outputs\\SNR_results.npy",SNR_results)
        plt.plot(1/noise_array,np.mean(SNR_results[:,0,:],axis=1))
        plt.plot(1/noise_array,np.mean(SNR_results[:,1,:],axis=1))
        plt.plot(1/noise_array,np.mean(SNR_results[:,2,:],axis=1))
        plt.plot(1/noise_array,np.mean(SNR_results[:,3,:],axis=1))
        plt.plot(1/noise_array,np.mean(SNR_results[:,4,:],axis=1))
        plt.plot(1/noise_array,np.mean(SNR_results[:,5,:],axis=1))
        plt.plot(1/noise_array,np.mean(SNR_results[:,6,:],axis=1))
        plt.legend(["physfad_c","zero order gradient descent","random search","dnn","physfad_c limited","zogd limited","random search limited"])
        # plt.legend(["random search","dnn","random search limited"])
        # plt.legend(["dnn 0.003","dnn 0.005","dnn 0.008","dnn 0.01"])
        plt.show()

    print("\nEnd Simulation")
if __name__ == "__main__":
    enclosure = {}
    # scipy.io.loadmat("..//PhysFad//ComplexEnclosure.mat",enclosure)
    main()
    # cProfile.run('main()')

