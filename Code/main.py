import numpy as np
import torch
import torch
from matplotlib import pyplot as plt
import scipy.io
import time
import random
import sys
# from memory_profiler import profile
from collections import deque
from models import *
from dataset import *
from conf import Config
import concurrent.futures
from tqdm import tqdm
from ChannelMatrixEvaluation import (test_configurations_capacity,test_configurations_capacity_serial,
                                       simulation_channel_optimization,
                                     zeroth_grad_optimization,random_search_optimization)
from PhysFadPy import physfad_c
from rate_model import capacity_loss
from benchmarks import benchmark_interface,zo_benchmark,random_benchmark,simulation_benchmark,simulation_noise_benchmark
import datetime
from functools import reduce
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import cProfile,pstats,io
from utils import (test_dnn_optimization, cosine_similarity, cosine_score, get_gradient_score, get_simulation_grads,
                   directional_derivative_accuracy, copy_with_gradients, copy_without_gradients, open_virtual_batch,
                   zo_estimate_gradient, save_fig,timeit,LimitedSizeDict)
from factories import abstract_factory, physfad_factory
# def breakdown_no_batch(ris_configuration, tx_x, tx_y,simulation,device,W=None):
#     H = simulation(ris_configuration, tx_x, tx_y,precalced_W=W)[0]
#
#     # if torch.any(~torch.isfinite(H)):
#     #     print("recognized non-finite value in simulation")
#     sigma = torch.tensor(1, dtype=torch.float64)
#     rate = capacity_loss(H, sigmaN=sigma, device=device,list_out=False).item()
#
#     return rate
# def breakdown(ris_configuration, tx_x, tx_y,simulation,device):
#     H = simulation(ris_configuration, tx_x, tx_y)[0]
#     # if torch.any(~torch.isfinite(H)):
#     #     print("recognized non-finite value in simulation")
#     sigma = torch.tensor(1, dtype=torch.float64)
#     rate_list = capacity_loss(H, sigmaN=sigma, device=device,list_out=False).item()
#     # rate_list = [rate.item() for rate in rate_list]
#     return rate_list

@timeit
def annealed_langevin_v3(model,simulation,starting_configuration,sow, number_of_iterations = 5,epsilon=torch.tensor(0.01),device=torch.device("cpu")):
    state_before_ald = model.training
    model.eval()
    # sigma_list = torch.linspace(1,0.01,10)
    sigma_list = torch.logspace(0,-2,10) # we use a geometric series for sigma values.

    batch_size = starting_configuration.shape[0]
    ris_configuration = starting_configuration
    configuration_list = []
    print(sigma_list)
    tau_vec = epsilon*(sigma_list**2)
    print("score parameter", tau_vec/(sigma_list**2))
    print("noise parameter", torch.sqrt(2*tau_vec))
    for i,sigma_i in enumerate(sigma_list):
        # rate = capacity_loss(simulation(ris_configuration, tx_x, tx_y), sigmaN=torch.tensor(1, dtype=torch.float64), device=device).item()
        # rate = breakdown(ris_configuration, tx_x, tx_y,simulation,device)
        # configuration_list.append((i*number_of_iterations, ris_configuration))

        for k in range(number_of_iterations):
            configuration_list.append((i * number_of_iterations + k, ris_configuration))
            sigma_v = sigma_i*torch.ones([batch_size,1],device=device,dtype=torch.float64)
            sow_repeated = sow[0].repeat(batch_size,1)
            normal_noise = torch.randn_like(ris_configuration)
            denoiser_output = model(ris_configuration,torch.hstack([sow_repeated,sigma_v]))
            score_function = denoiser_output - ris_configuration
            ris_configuration = ris_configuration + (tau_vec[i]/sigma_list[i]**2) * score_function + torch.sqrt(2*tau_vec[i])*normal_noise

            ris_configuration = torch.clip(ris_configuration,0,1)


        # config_norm = torch.norm(ris_configuration).item()
        # print("alpha",sigma_i.item(),"k",k,"rate",rate,"norm",config_norm)
    model.training = state_before_ald
    return configuration_list

def diffusion_training_score_function(model_diffusion,test_ldr,optimizer_diffusion, simulation,config, device="cpu"):

    # batch_size=16
    batch_size = int(config.batch_size//2)
    # x_tx_orig = torch.tensor([0, 0, 0]).repeat(batch_size,1).to(device)
    # y_tx_orig = torch.tensor([4, 4.5, 5]).repeat(batch_size,1).to(device)
    capacity_simulation = lambda x,tx_x,tx_y: -capacity_loss(simulation(x, tx_x, tx_y), sigmaN=torch.tensor(1,dtype=torch.float64),list_out=True, device=device)
    sigma_min=0.01
    sigma_max=1
    q=deque(maxlen=10)
    for i in range(100000):
        # generate batch
        X = torch.rand([batch_size, config.input_size], device=device,dtype=torch.float64)
        # tx_x_diff = 19.5 * torch.rand([batch_size, 3], device=device,dtype=torch.float64) - 3.3
        # tx_y_diff = 11.5 * torch.rand([batch_size, 3], device=device,dtype=torch.float64) - 2.8

        # tx_x, tx_y = x_tx_orig + tx_x_diff, y_tx_orig + tx_y_diff
        tx_x, tx_y = simulation.generate_tx_location(batch_size,device)
        sigma = torch.FloatTensor(batch_size,1).uniform_(sigma_min, sigma_max).to(device).type(torch.float64)

        optimizer_diffusion.zero_grad()
        # Estimation of the gradient of the log of the probability function
        score_function = model_diffusion(torch.hstack([X,tx_x/20,tx_y/20,sigma]))
        # get the denoised configuration using the score function
        improved_X = torch.clip((sigma**2)*score_function+X,0,1)

        # Sanity checks for the data
        print("input std: ", X.std(dim=0).mean().item())
        print("output std: ", improved_X.std(dim=0).mean().item())
        capacity_before = test_configurations_capacity(simulation,X,tx_x,tx_y,device,list_out=False,noise=None)[0]
        capacity_after = test_configurations_capacity(simulation,improved_X,tx_x,tx_y,device,list_out=False,noise=None)[0]
        print("capacity before: {0} and after: {1}".format(capacity_before.item(),capacity_after.item()))
        sys.stdout.flush()

        # get the estimate of the gradient for the denoiser
        epsilon = 1  # 0.0001
        grad_inp_64 = zo_estimate_gradient(capacity_simulation, improved_X , tx_x, tx_y, epsilon, 64, device,broadcast_tx=False)

        # sigma regularization
        x_distance = ((improved_X - X) ** 2).sum(dim=1) / config.input_size
        sigma_distance = ((x_distance - sigma.squeeze(1)) ** 2).mean()
        sigma_grad = torch.autograd.grad(sigma_distance, improved_X, retain_graph=True, create_graph=True)[0]
        q.append(abs(sigma_distance).sum().item())
        print(sum(q) / len(q)) # check if the sigma error is getting smaller on average


        grad_inp = grad_inp_64

        total_grad = grad_inp+sigma_grad

        improved_X.backward(total_grad)
        optimizer_diffusion.step()
        if i % 60 == 0 and i != 0:
            model_diffusion.eval()
            torch.save(model_diffusion.state_dict(), "./Models/Full_Main_model2.pt")
            acc_rate = 0
            # print(len(test_ldr))
            ald_capacity_avg = np.zeros(10)
            ald_iter_list = np.zeros(10)
            zogd_capacity_avg = np.zeros(50)
            simulation_capacity_avg = np.zeros(50)
            for batch in test_ldr:
                (X, X_gradients, tx_x, tx_y, Y_capacity, Y) = open_virtual_batch(batch)
                X = X.type(torch.float64)
                X = X[0].unsqueeze(0)
                improved_X = model_diffusion(torch.hstack([X,tx_x, tx_y,torch.tensor(1,device=device).unsqueeze(0).unsqueeze(0)]))
                ald_results = annealed_langevin_v3(model_diffusion, simulation, X, tx_x, tx_y, epsilon=config.diffusion_epsilon,
                                                device=device)
                # ald_iters = list(ald_iters)
                # ald_capac = list(ald_capac)
                for i,(iter,capac) in enumerate(ald_results):
                    ald_capacity_avg[i] += capac
                    ald_iter_list[i] = iter

                (zogd_time_lst, zogd_capacity, zogd_gradient_score) = zeroth_grad_optimization(device, simulation,
                                                                                               X.clone().detach().requires_grad_(
                                                                                                   False).to(
                                                                                                   device), tx_x,
                                                                                               # change require grad to false
                                                                                               tx_y,
                                                                                               noise_power=1,
                                                                                               num_of_iterations=50)
                for i,capac in enumerate(zogd_capacity):
                    zogd_capacity_avg[i] += capac

                rate, H = test_configurations_capacity(simulation, improved_X, tx_x, tx_y, device, list_out=False,
                                                       noise=None)
                acc_rate = acc_rate+rate

            print("test diffusion mean rate: ",(acc_rate/len(test_ldr)).item())
            plt.plot(ald_iter_list,ald_capacity_avg/len(test_ldr))
            plt.plot(zogd_capacity_avg/len(test_ldr))
            plt.legend(["ald", "zogd"])
            plt.title("capacity at iteration for different algorithms") # average results over a small test set of 8 tx locations
            save_fig("diffusion_capacity_per_iteration_"+str(i)+".pdf","plots")
            plt.show()
        model_diffusion.train()

def diffusion_training(model_forward,model_diffusion,train_ldr,test_ldr,optimizer_diffusion, simulation,config, max_epochs, ep_log_interval, output_size, output_shape,
                       model_output_capacity, device="cpu"):

    batch_size=16
    capacity_simulation = lambda x,tx_x,tx_y: -capacity_loss(simulation(x, tx_x, tx_y), sigmaN=torch.tensor(1,dtype=torch.float64),list_out=True, device=device)
    sigma_min=0.01
    sigma_max=1
    q=deque(maxlen=10)
    for i in range(100000):

        X = torch.rand([batch_size, config.input_size], device=device,dtype=torch.float64)

        tx_x, tx_y = simulation.generate_tx_location(batch_size,device)
        sigma = torch.FloatTensor(batch_size,1).uniform_(sigma_min, sigma_max).to(device).type(torch.float64)
        optimizer_diffusion.zero_grad()
        improved_X = model_diffusion(torch.hstack([X,tx_x/20,tx_y/20,sigma]))
        print("input std: ", X.std(dim=0).mean().item())
        print("output std: ", improved_X.std(dim=0).mean().item())
        epsilon = 1 # 0.0001
        x_distance = ((improved_X - X) ** 2).sum(dim=1)/config.input_size
        sigma_distance = ((x_distance - sigma.squeeze(1)) ** 2).mean()
        sigma_grad = torch.autograd.grad(sigma_distance, improved_X, retain_graph=True, create_graph=True)[0]
        acc_gradients = get_simulation_grads(improved_X, tx_x, tx_y, simulation, device, noise=None,broadcast_tx=False)



        capacity_before = test_configurations_capacity(simulation,X,tx_x,tx_y,device,list_out=False,noise=None)[0]
        capacity_after = test_configurations_capacity(simulation,improved_X,tx_x,tx_y,device,list_out=False,noise=None)[0]

        print("capacity before: {0} and after: {1}".format(capacity_before.item(),capacity_after.item()))
        sys.stdout.flush()

        grad_inp_64 = zo_estimate_gradient(capacity_simulation, improved_X , tx_x, tx_y, epsilon, 64, device,broadcast_tx=False)

        grad_inp = grad_inp_64

        q.append(abs(sigma_distance).sum().item())
        print(sum(q)/len(q))
        total_grad = grad_inp+sigma_grad

        improved_X.backward(total_grad)
        optimizer_diffusion.step()
        if i % 60 == 0 and i != 0:
            model_diffusion.eval()
            torch.save(model_diffusion.state_dict(), "./Models/Full_Main_model2.pt")
            acc_rate = 0
            ald_capacity_avg = np.zeros(10)
            ald_iter_list = np.zeros(10)
            zogd_capacity_avg = np.zeros(50)
            simulation_capacity_avg = np.zeros(50)
            for batch in test_ldr:
                (X, X_gradients, tx_x, tx_y, Y_capacity, Y) = open_virtual_batch(batch)
                X = X.type(torch.float64)
                X = X[0].unsqueeze(0)
                improved_X = model_diffusion(torch.hstack([X,tx_x, tx_y,torch.tensor(1,device=device).unsqueeze(0).unsqueeze(0)]))
                ald_results = annealed_langevin_v3(model_diffusion, simulation, X, tx_x, tx_y, epsilon=config.diffusion_epsilon,
                                                device=device)

                for i,(iter,capac) in enumerate(ald_results):
                    ald_capacity_avg[i] += capac
                    ald_iter_list[i] = iter

                (zogd_time_lst, zogd_capacity, zogd_gradient_score) = zeroth_grad_optimization(device, simulation,
                                                                                               X.clone().detach().requires_grad_(
                                                                                                   False).to(
                                                                                                   device), tx_x,
                                                                                               # change require grad to false
                                                                                               tx_y,
                                                                                               noise_power=1,
                                                                                               num_of_iterations=50)
                for i,capac in enumerate(zogd_capacity):
                    zogd_capacity_avg[i] += capac

                rate, H = test_configurations_capacity(simulation, improved_X, tx_x, tx_y, device, list_out=False,
                                                       noise=None)
                acc_rate = acc_rate+rate
            print("test diffusion mean rate: ",(acc_rate/len(test_ldr)).item())
            plt.plot(ald_iter_list,ald_capacity_avg/len(test_ldr))
            plt.plot(zogd_capacity_avg/len(test_ldr))
            # plt.plot(simulation_capacity_avg/len(test_ldr))
            plt.legend(["ald", "zogd"])
            plt.title("capacity at iteration for different algorithms") # average results over a small test set of 8 tx locations
            # plt.show()
            save_fig("diffusion_capacity_per_iteration_"+str(i)+".pdf","plots")
            plt.show()
        model_diffusion.train()
        # print(rate.item())


def diffusion_active_training(model_diffusion, train_ldr:abstract_dataset, test_ldr,optimizer_diffusion, simulation, config, device=torch.device("cpu")):

    batch_size=int(config.batch_size//2) # faster training
    # simulation_lmbda = lambda x,sow: simulation(x, sow,snr_noise=torch.tensor(1,dtype=torch.float64),list_out=True)
    sigma_min = config.diffusion_min_sigma
    sigma_max = config.diffusion_max_sigma
    epsilon = 1  # 0.0001
    zo_gradient_approximation = zo_benchmark("zo_gradient_approximation", simulation, device,m=64,epsilon = epsilon,broadcast_sow=False)
    capacity_physfad = lambda x,sow: -capacity_loss(simulation(x, sow)[1], sigmaN=torch.tensor(1,dtype=torch.float64),list_out=True, device=device)

    q=deque(maxlen=10)
    active_training_memory = LimitedSizeDict(size_limit=128)
    for i in range(1000):
        if random.random() < config.new_configuration_chance or len(active_training_memory)==0: # TODO: This should be handled with a dataset
            # X = torch.rand([batch_size, config.input_size], device=device,dtype=torch.float64)
            (X,sow) = train_ldr.dataset.generate_batch(batch_size,device)
            # tx_x,tx_y = sow[:,0:3],sow[:,3:6]
            # tx_x, tx_y = simulation.generate_tx_location(batch_size,device)
            # sow = [tx_x,tx_y]
            sigma = torch.FloatTensor(batch_size,1).uniform_(sigma_min, sigma_max).to(device).type(torch.float64)
            iter_count = 0
        else: # choose from memory one of the previous 128 iterations
            key = random.choice(list(active_training_memory.keys()))
            X = active_training_memory[key]["ris_config"]
            sow = active_training_memory[key]["SoW"]
            sigma = torch.clip(active_training_memory[key]["sigma"]*9/10,sigma_min,sigma_max)
            iter_count = active_training_memory[key]["iteration_count"]
            print("learning from previous iteration: ", key," repeat count: ",iter_count,"sigma values: ", sigma)
        optimizer_diffusion.zero_grad()
        improved_X = model_diffusion(X,torch.hstack([sow,sigma])) # TODO: add SNR
        # improved_X = model_diffusion(torch.hstack([X,sow,sigma]))
        if iter_count < 20:
            active_training_memory[i] = {"ris_config":copy_without_gradients(improved_X),"SoW":sow,"sigma":sigma,"iteration_count":iter_count+1}

        # sanity checks and live improvement updates
        print("input std: ", X.std(dim=0).mean().item())
        print("output std: ", improved_X.std(dim=0).mean().item())
        capacity_before = simulation(X,sow,list_out=False,snr_noise=None)[0]
        print("just finished the before")
        capacity_after = simulation(improved_X,sow,list_out=False,snr_noise=None)[0]
        print("capacity before: {0} and after: {1}".format(capacity_before.item(),capacity_after.item()))
        sys.stdout.flush()

        # sigma regularization
        x_distance = ((improved_X - X) ** 2).mean(dim=1)
        # sigma_distance = ((x_distance - sigma.squeeze(1)) ** 2).mean()
        sigma_loss = ((1/sigma**2)*x_distance).mean()
        sigma_grad = torch.autograd.grad(sigma_loss, improved_X, retain_graph=True, create_graph=True)[0]


        # get the estimate of the gradient of the rate with respect to the configuration
        # grad_inp_16 = zo_estimate_gradient(capacity_simulation, improved_X , tx_x, tx_y, epsilon, 16, device,broadcast_tx=False)
        # grad_inp_64 = zo_estimate_gradient(capacity_simulation, improved_X , tx_x, tx_y, epsilon, 64, device,broadcast_tx=False)
        approx_grad_inp_64 = zo_gradient_approximation.loss_and_gradient(improved_X,sow,snr_noise=None)[1]
        # approx_grad_inp_64_old = zo_estimate_gradient(capacity_physfad, improved_X , sow, epsilon, 64, device,broadcast_tx=False)
        print("finished the gradient approximation")
        # exit()
        # grad_inp_64 = get_simulation_grads(copy_with_gradients(improved_X), sow, simulation, device, noise=None,broadcast_tx=False)
        # print("cosine score (approx new, acc): ", cosine_score(approx_grad_inp_64,grad_inp_64))
        # print("cosine score (approx old, acc): ", cosine_score(approx_grad_inp_64_old,grad_inp_64))
        # print("cosine score (approx old, approx new): ", cosine_score(approx_grad_inp_64,approx_grad_inp_64_old))
        grad_inp = approx_grad_inp_64

        # the total gradient with respect to the configuration
        total_grad = grad_inp+config.lmbda*sigma_grad

        # backpropagation using gradients and optimizer step
        improved_X.backward(total_grad)
        optimizer_diffusion.step()
        if i % config.ep_log_interval == 0 and i != 0:
            model_diffusion.eval()
            torch.save(model_diffusion.state_dict(), "./Models/Full_Main_model5.pt")
            # acc_rate = 0
            # ald_capacity_avg = np.zeros([10,batch_size])
            # ald_iter_list = np.zeros(10)
            # zogd_capacity_avg = np.zeros(50)
            # simulation_capacity_avg = np.zeros(50)
            # for batch in test_ldr:
            #     (X, X_gradients, tx_x, tx_y, Y_capacity, Y) = open_virtual_batch(batch)
            #     X = X.type(torch.float64)
            #     X = X[0:batch_size]
            #     # improved_X = model_diffusion(torch.hstack([X,tx_x, tx_y,torch.tensor(1,device=device).unsqueeze(0).unsqueeze(0)]))
            #     ald_configuration_results = annealed_langevin_v3(model_diffusion, simulation, X, tx_x, tx_y, epsilon=config.diffusion_epsilon,
            #                                                      device=device)
            #
            #     for i, (iter, ris_configuration) in enumerate(ald_configuration_results):
            #         capac, _ = test_configurations_capacity(simulation, ris_configuration, tx_x, tx_y, device, list_out=False)
            #         # capac = breakdown(ris_configuration, tx_x, tx_y, simulation, device)
            #         # for i,(iter,capac) in enumerate(ald_results):
            #         ald_capacity_avg[i] += capac
            #         ald_iter_list[i] = iter
            #
            #     (zogd_time_lst, zogd_capacity, zogd_gradient_score) = zeroth_grad_optimization(device, simulation,
            #                                                                                    X.clone().detach().requires_grad_(
            #                                                                                        False).to(
            #                                                                                        device), tx_x,
            #                                                                                    # change require grad to false
            #                                                                                    tx_y,
            #                                                                                    noise_power=1,
            #                                                                                    num_of_iterations=2)
            #     for i,capac in enumerate(zogd_capacity):
            #         zogd_capacity_avg[i] += capac
            #     # (simulation_time_lst, simulation_capacity, simulation_inputs) = (
            #     #     simulation_channel_optimization(device, simulation,copy_with_gradients(X,device),tx_x, tx_y,noise_power=1,
            #     #                                  learning_rate=0.1,num_of_iterations=2))  # 50
            #     # for i,capac in enumerate(simulation_capacity):
            #     #     simulation_capacity_avg[i] += capac
            #
            #     # H_approx = model_forward(tx_x, tx_y,improved_X)
            #     # rate_approx = capacity_loss(H_approx.reshape(H_approx.shape[0], output_size, output_shape[0], output_shape[1]), list_out=True,
            #     #               device=device)
            #     rate, H = test_configurations_capacity(simulation, improved_X, tx_x, tx_y, device, list_out=False,
            #                                            noise=None)
            #     acc_rate = acc_rate+rate
            #     # H_test_mse = ((abs(H).reshape([H_approx.shape[0],-1])-H_approx)**2).mean()/abs((H**2).mean())
            # # print("test diffusion mean rate: ",(acc_rate/len(test_ldr)).item(),"test forward model accuracy(NMSE): ", H_test_mse.item())
            # print("test diffusion mean rate: ",(acc_rate/len(test_ldr)).item())
            # plt.plot(ald_iter_list,ald_capacity_avg/len(test_ldr))
            #
            # for i in range(len(ald_capacity_avg[:, 0])):
            #     ald_capacity_avg[i, 0] = ald_capacity_avg[i, 1:].mean()
            # plt.plot(zogd_capacity_avg/len(test_ldr), linewidth=4.0, linestyle='dashed')
            # plt.plot(ald_iter_list, ald_capacity_avg[:, 0] / len(test_ldr), linewidth=4.0, linestyle='dashed')
            # # plt.plot(ald_iter_list, ald_capacity_avg[:, 1:] / len(test_ldr), alpha=0.5)
            #
            # # plt.plot(simulation_capacity_avg/len(test_ldr))
            # plt.legend(["zogd", "ald Mean","ald batch"])
            # plt.title("capacity at iteration for different algorithms") # average results over a small test set of 8 tx locations
            # # plt.show()
            # save_fig("diffusion_capacity_per_iteration_"+str(i)+".pdf","plots")
            # plt.show()
        model_diffusion.train()
        # print(rate.item())
def diffusion_inference(model_diffusion,test_ldr, simulation, config, device=torch.device("cpu")):
    output_graph = np.zeros([5,50])
    ald_capacity_avg = np.zeros(50)
    ald_iter_list = np.zeros(50)
    zogd_capacity_avg = np.zeros(50)
    simulation_capacity_avg = np.zeros(50)
    simulation_capacity_avg_with_noise = np.zeros(50)
    for batch_idx,batch in enumerate(test_ldr):
        if batch_idx > 4:
            batch_idx = batch_idx-1
            break
        print(batch_idx)
        # (X, sow) = open_virtual_batch(batch)
        (X, sow) = test_ldr.dataset.generate_batch(config.batch_size, device)
        X = X.type(torch.float64)
        X = X[0:8] # Speeds up the run
        sow = sow[0:8] # Speeds up the run
        simulation.plot_environment(sow)

        ## Testing the diffusion model
        ald_configuration_results = annealed_langevin_v3(model_diffusion, simulation, X, sow,epsilon=config.diffusion_epsilon, device=device)
        for i, (iter, ris_configuration) in enumerate(ald_configuration_results): # TODO: use this code
            capac,_ = simulation(ris_configuration, sow, list_out=False)
            # capac = breakdown(ris_configuration,tx_x,tx_y,simulation,device)
            print(i,capac)
            ald_capacity_avg[i] += capac
            output_graph[0,i] += capac
            ald_iter_list[i] = iter
        benchmarks_list = []
        #zo_benchmark,random_benchmark,simulation_benchmark,simulation_noise_benchmark
        # TODO: might want to add constants/ configurations for these:
        benchmarks_list.append(zo_benchmark("zo_gradient_approximation", simulation, device,learning_rate=0.1, broadcast_sow=False))
        benchmarks_list.append(random_benchmark("random_search", simulation, device))
        benchmarks_list.append(simulation_benchmark("simulation_channel_optimization", simulation, device,learning_rate=0.1))
        benchmarks_list.append(simulation_noise_benchmark("simulation_channel_optimization_noise", simulation, device,simulation_benchmark_without_noise=benchmarks_list[-1]))
        benchmark_results = {}
        for i, benchmark in enumerate(benchmarks_list):
            benchmark_results[benchmark.name] = benchmark.run(X,sow, num_of_iterations =50, snr_noise = 1)
            output_graph[i+1] += np.array(benchmark_results[benchmark.name][1])

        simulation.save_and_change_to_clean_environment()
        simulation.plot_environment(sow)


        print(simulation.current_state)
        print("done")
        # for key, value in benchmark_results.items():
        #     plt.plot(value[1])
        # plt.plot(output_graph[0])
        # To show the results with a cleaner plot, run plot_iteration_graph.py after this
        linestyle = ['-*', '-*', '-*', '-*', '-*']
        linewidth = [2, 1.5, 1.5, 1.5, 1.5]
        for i in range(len(output_graph)):
            plt.plot(output_graph[i]/(120*(batch_idx+1)),linestyle[i], linewidth=linewidth[i],
                     label=benchmarks_list[i-1].name if i > 0 else "Diffusion Model")

        plt.legend([benchmarks_list[i-1].name if i >0 else "Diffusion Model" for i in range(len(output_graph))])
        plt.title("capacity at iteration for different algorithms")
        plt.xlabel("iteration")
        plt.ylabel("Rate [Bits/Channel use]")
        plt.grid(visible=True)
        # save_fig("diffusion_capacity_per_iteration_" + str(batch_idx) + ".pdf", "plots")
        plt.show()
        ## Testing simulation on clean environment
        # simulation.save_and_change_to_clean_environment() # Load the clean environment and reset the bessel memory
        # (_, simulation_capacity, _, simulation_ris_configurations) = (
        #     simulation_channel_optimization(device, simulation, copy_with_gradients(torch.special.logit(X), device), sow, noise_power=1,
        #                                  learning_rate=0.1, num_of_iterations=50))  # 50
        # for i, capac in enumerate(simulation_capacity):
        #     simulation_capacity_avg[i] += capac
        #     output_graph[1,i] += capac


        ## Testing the same simulation configurations on the noisy environment
        # simulation.reload_original_environment() # Load the original environment and reset the bessel memory
        # for i, capac in enumerate(simulation_capacity):
        #     capacity,_ = test_configurations_capacity(simulation, simulation_ris_configurations[i], tx_x, tx_y, device, list_out=False)
        #     print("i: ", i,"simulation+noise capacity: ", capacity)
        #     simulation_capacity_avg_with_noise[i] += capacity.detach().numpy()
        #     output_graph[2,i] += capacity.detach().numpy()


        ## Testing ZOGD
        # (zogd_time_lst, zogd_capacity, _) = zeroth_grad_optimization(device, simulation,
        #                                                              copy_without_gradients(X,device), tx_x, tx_y,
        #                                                              noise_power=1,
        #                                                              num_of_iterations=50)
        # for i, capac in enumerate(zogd_capacity):
        #     zogd_capacity_avg[i] += capac
        #     output_graph[3, i] += capac


    # np.save("./outputs/iteration_graph_results.npy", output_graph) # save the results


    # plt.plot(simulation_capacity_avg/(120*(batch_idx+1)),'-*',linewidth=1.5)#,linestyle='dashed')
    # plt.plot(simulation_capacity_avg_with_noise/(120*(batch_idx+1)),'-*',linewidth=1.5)#,linestyle='dashed')
    # plt.plot(zogd_capacity_avg / (120*(batch_idx+1)),'-*',linewidth=1.5)#,linestyle='dashed')
    # plt.plot(ald_iter_list, ald_capacity_avg / (120*(batch_idx+1)),'-*',linewidth=2)#,linestyle='dashed')
    # plt.grid(visible=True)

    # plt.legend(["Simulation GD","Simulation GD + 0.01 noise on env","ZOGD","ZO-ALD"])
    # plt.title(
    #     "capacity at iteration for different algorithms")  # average results over a small test set of 8 tx locations
    # # plt.show()
    # plt.xlabel("iteration")
    # plt.ylabel("Rate [Bits/Channel use]")
    # save_fig("diffusion_capacity_per_iteration_" + str(i) + ".pdf", "plots")
    # plt.show()


def generate_dataset(dataset_name,dataset_post_name,dataset_path,virt_batch_size,dataset_size,simulation,input_size,device):
    X = torch.rand([dataset_size, input_size], device=device, dtype=torch.float64)
    tx_x, tx_y = simulation.generate_tx_location(int(dataset_size / virt_batch_size),device)
    rate, H = test_configurations_capacity(simulation, X[0:virt_batch_size], tx_x[0].unsqueeze(0), tx_y[0].unsqueeze(0), device, list_out=False,
                                           noise=None)
    rate_dataset = torch.zeros(dataset_size)
    H_dataset = torch.zeros([dataset_size,*H.shape[1:]],dtype=torch.complex64)

    for batch_idx in range(int(dataset_size / virt_batch_size)):
        print("generating batches", dataset_post_name,": ", 100 * batch_idx * virt_batch_size / dataset_size, "%")
        rate, H = test_configurations_capacity(simulation, X[virt_batch_size * batch_idx:virt_batch_size * (batch_idx + 1)], tx_x[batch_idx].unsqueeze(0), tx_y[batch_idx].unsqueeze(0), device, list_out=True, noise=None)
        rate_dataset[virt_batch_size * batch_idx:virt_batch_size * (batch_idx + 1)] = rate
        H_dataset[virt_batch_size * batch_idx:virt_batch_size * (batch_idx + 1)] = H

    ris_configuration_file_name = dataset_path+dataset_name+"_RISConfiguration"+dataset_post_name+".mat"
    H_mat_file_name = dataset_path+dataset_name+"_H_realizations"+dataset_post_name+".mat"
    rate_file_name = dataset_path+dataset_name+"_H_capacity"+dataset_post_name+".mat"
    tx_file_name = dataset_path+dataset_name+"_transmitter_location"+dataset_post_name+".mat"

    scipy.io.savemat(ris_configuration_file_name,{"RISConfiguration":X.detach().numpy()})
    scipy.io.savemat(H_mat_file_name,{"sampled_Hs":H_dataset.detach().numpy()})
    scipy.io.savemat(rate_file_name,{"rate":rate_dataset.detach().numpy()})
    scipy.io.savemat(tx_file_name,{"x_tx_modified":tx_x.detach().numpy(),"y_tx_modified":tx_y.detach().numpy()})

def load_data(factory,batch_size,config,device):
    train_ds = factory.create_dataset(config,batch_size,device)
    test_ds = factory.create_dataset(config,batch_size,device)
    train_ldr = train_ds.load("")
    test_ldr = test_ds.load("_test")

    return train_ds,test_ds,train_ldr,test_ldr


def room_visualization(net_diffusion, test_ldr, simulation, config, device, optimized=True):
    x_sample_density = 26 * 5
    y_sample_density = 22 * 5
    # locations
    x_rx_orig = torch.tensor([15, 15, 15, 15]).to(device)
    y_rx_orig = torch.tensor([11, 11.5, 12, 12.5]).to(device)
    x_diff = torch.linspace(-20, 5, x_sample_density).unsqueeze(1).repeat(1, y_sample_density)
    y_diff = torch.linspace(-14.5, 6.5, y_sample_density).repeat(x_sample_density, 1)
    (X, _, _, _, _, _) = open_virtual_batch(next(iter(test_ldr)))  # (predictors, targets)
    X = X[0].unsqueeze(0)
    tx_x = torch.tensor([0, 0, 0]).unsqueeze(0).to(device)
    tx_y = torch.tensor([4, 4.5, 5]).unsqueeze(0).to(device)
    if optimized:
        ald_configuration_results = annealed_langevin_v3(net_diffusion, simulation,
                                                         copy_without_gradients(X.type(torch.float64), device),
                                                         tx_x, tx_y, epsilon=config.diffusion_epsilon, device=device)
        ris_configuration = ald_configuration_results[-1][1]
    else:
        ris_configuration = X
    room_image = torch.zeros([x_sample_density, y_sample_density])



    for i in range(x_sample_density):
        list_of_locations = []
        for j in range(y_sample_density):
            rx_x, rx_y = (x_rx_orig + x_diff[i, j]).unsqueeze(0), (y_rx_orig + y_diff[i, j]).unsqueeze(0)
            list_of_locations.append((rx_x, rx_y))

        with concurrent.futures.ProcessPoolExecutor() as executer:
            results = list(tqdm(executer.map(simulation.get_bessel_w_rx_change, list_of_locations),
                                total=len(list_of_locations)))

        idx = 0
        W_list = []
        for W in results:
            W_list.append(W)
        for j in range(y_sample_density):
            print(i, j)
            rx_x, rx_y = (x_rx_orig + x_diff[i, j]).unsqueeze(0), (y_rx_orig + y_diff[i, j]).unsqueeze(0)
            simulation.change_rx_location(rx_x, rx_y)

            room_image[i, j] = breakdown_no_batch(ris_configuration, tx_x, tx_y, simulation, device, W_list[idx])
            room_image[i, j] = test_configurations_capacity(simulation, ris_configuration, tx_x, tx_y, device, list_out=False)[0]
            idx = idx + 1

    print("done")
    print(room_image)


    if optimized:
        torch.save(room_image,"plots\\optimized_room.pt")
        save_fig("optimized_room.pdf", "plots")
    else:
        torch.save(room_image, "plots\\unoptimized_room.pt")
        save_fig("unoptimized_room.pdf", "plots")
    plt.imshow(room_image)
    plt.show()
def room_graph_print(simulation):
    unoptimized = torch.load("plots\\unoptimized_room.pt")
    optimized = torch.load("plots\\optimized_room.pt")
    # (freq, x_tx, y_tx, fres_tx, chi_tx, gamma_tx,
    #  x_rx, y_rx, fres_rx, chi_rx, gamma_rx,
    #  x_env, y_env, fres_env, chi_env, gamma_env, x_ris_c, y_ris_c) = simulation.parameters

    plt.imshow((unoptimized).T, cmap='cividis')
    plt.plot((simulation.parameters["x_ris"][0] + 5) * 5.2, (simulation.parameters["y_ris"][0] + 2.5) * 5.238, 'o', fillstyle="none", color="yellow")
    plt.plot((simulation.parameters["x_tx"][0] + 5) * 5.2, (simulation.parameters["y_tx"][0] + 2.5) * 5.238, 'o', fillstyle="none", color="orange")
    plt.plot((simulation.parameters["x_rx"][0] + 5) * 5.2, (simulation.parameters["y_rx"][0] + 2.5) * 5.238, 'o', fillstyle="none", color="red")
    simulation.plot_environment(x_scaler=5.2, y_scaler=5.238, x_shift=5, y_shift=2.5)
    plt.imshow((optimized).T, cmap='cividis')
    plt.plot((simulation.parameters["x_ris"][0] + 5) * 5.2, (simulation.parameters["y_ris"][0] + 2.5) * 5.238, 'o', fillstyle="none", color="yellow")
    plt.plot((simulation.parameters["x_tx"][0] + 5) * 5.2, (simulation.parameters["y_tx"][0] + 2.5) * 5.238, 'o', fillstyle="none", color="orange")
    plt.plot((simulation.parameters["x_rx"][0] + 5) * 5.2, (simulation.parameters["y_rx"][0] + 2.5) * 5.238, 'o', fillstyle="none", color="red")
    simulation.plot_environment(x_scaler=5.2, y_scaler=5.238, x_shift=5, y_shift=2.5)

    fig1, ax1 = plt.subplots()

    ax1.imshow((unoptimized - optimized).T, cmap='cividis')
    ax1.plot((simulation.parameters["x_ris"][0] + 5) * 5.2, (simulation.parameters["y_ris"][0] + 2.5) * 5.238, 'o', fillstyle="none", color="yellow")
    ax1.plot((simulation.parameters["x_tx"][0] + 5) * 5.2, (simulation.parameters["y_tx"][0] + 2.5) * 5.238, 'o', fillstyle="none", color="orange")
    ax1.plot((simulation.parameters["x_rx"][0] + 5) * 5.2, (simulation.parameters["y_rx"][0] + 2.5) * 5.238, 'o', fillstyle="none", color="red")
    simulation.plot_environment(x_scaler=5.2, y_scaler=5.238, x_shift=5, y_shift=2.5,show=False)
    x1, x2, y1, y2 = 95, 110, 55, 85  # subregion of the original image
    axins = ax1.inset_axes(
        [0.1, 0.03, 0.4, 0.4],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.imshow((unoptimized - optimized).T, cmap='cividis')
    axins.plot((simulation.parameters["x_rx"][0] + 5) * 5.2, (simulation.parameters["y_rx"][0] + 2.5) * 5.238, 'o', fillstyle="none", color="red")
    axins.set_xlim(x1, x2)  # apply the x-limits
    axins.set_ylim(y2, y1)  # apply the y-limits
    _patch, pp1, pp2 = mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="red")
    pp1.loc1, pp1.loc2 = 1, 4  # inset corner 1 to origin corner 4 (would expect 1)
    pp2.loc1, pp2.loc2 = 4, 1  # inset corner 3 to origin corner 2 (would expect 3)
    plt.show()
def find_optimal_learning_rate(simulation,test_ldr,config,device):
    device_cpu = torch.device('cpu')


    (X, _, _, _, _, _) = open_virtual_batch(next(iter(test_ldr)))  # (predictors, targets)
    # we test the learning rate on the default tx location
    tx_x, tx_y = simulation.generate_tx_location(1, device, modify=False)

    initial_inp = X[0, :].unsqueeze(0).clone().detach().requires_grad_(True).to(device_cpu)
    learning_rates = np.arange(0.1, 0.25, 0.05)  # learning rates tested
    num_of_iter = 50
    num_of_lr = len(learning_rates)
    capacities_curves = np.zeros([num_of_lr, num_of_iter])
    for i, lr in enumerate(learning_rates):
        (simulation_time_lst, simulation_capacity, simulation_inputs) = simulation_channel_optimization(device_cpu, simulation,
                                                                                            copy_with_gradients(
                                                                                                initial_inp,
                                                                                                device_cpu), tx_x, tx_y,
                                                                                            noise_power=1,
                                                                                            learning_rate=lr,
                                                                                            num_of_iterations=num_of_iter)  # 50
        capacities_curves[i] = simulation_capacity
        plt.plot((simulation_inputs[-1]).detach().numpy())
        plt.show()
    plt.plot(capacities_curves.T)
    plt.legend([str(x) for x in learning_rates])
    plt.show()
def snr_graph(simulation, test_ldr, net_diffusion, config, device):
    device_cpu = torch.device('cpu')
    device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_snr_points = 11  # 15

    attempts = 6  # 6
    snr_values = np.linspace(0.5, 60, num_snr_points)
    # snr_values = np.linspace(0.8,1.2,num_snr_points)

    noise_array = 1 / snr_values
    SNR_results = np.zeros([num_snr_points, 7, attempts])
    for attempt in range(attempts):
        (X, _, tx_x, tx_y, _, _) = open_virtual_batch(next(iter(test_ldr)))  # (predictors, targets)
        X = X.type(torch.float64)
        # X = X[0].unsqueeze(0)
        X = X[0:8]  # Speed up the results
        for i, noise in enumerate(noise_array):
            print(noise, i, attempt)
            # initial_inp = X[0, :].unsqueeze(0).clone().detach().requires_grad_(True).to(device_cpu)
            initial_inp = X.clone().detach().requires_grad_(True).to(device_cpu)
            simulation.plot_environment(tx_x, tx_y)
            simulation.save_and_change_to_clean_environment()
            (simulation_time_lst, simulation_capacity, last_configuration, simulation_inputs) = simulation_channel_optimization(
                device_cpu, simulation,
                copy_with_gradients(
                    initial_inp,
                    device_cpu),
                tx_x, tx_y,
                noise_power=noise,
                learning_rate=0.1,
                num_of_iterations=100)  # 150
            simulation.reload_original_environment()
            simulation_capacity_avg_with_noise = torch.zeros(len(simulation_capacity))
            for j, capac in enumerate(simulation_capacity):
                simulation_capacity_avg_with_noise[j], _ = test_configurations_capacity(simulation, simulation_inputs[j], tx_x,
                                                                                     tx_y, device,
                                                                                     list_out=False, noise=noise)
                print("i: ", j, "simulation+noise capacity: ", simulation_capacity_avg_with_noise[j])

            (zogd_time_lst, zogd_capacity, _) = zeroth_grad_optimization(device, simulation,
                                                                         copy_without_gradients(X, device), tx_x,
                                                                         tx_y, noise_power=noise,
                                                                         num_of_iterations=100)  # 200

            (rand_search_time_lst, random_search_capacity) = random_search_optimization(simulation, 100, device_cpu,
                                                                                        noise_power=noise, tx_x=tx_x,
                                                                                        tx_y=tx_y)
            ald_configuration_results = annealed_langevin_v3(net_diffusion, simulation,
                                                             copy_without_gradients(initial_inp.type(torch.float64),
                                                                                    device_cpu), tx_x,
                                                             tx_y, epsilon=5 * 10 ** (-1), device=device)

            dnn_simulation_capacity_lst = test_dnn_optimization(simulation, [x[1].cpu() for x in ald_configuration_results],
                                                             tx_x, tx_y, device_cpu, noise=noise)

            simulation_capacity = np.array(simulation_capacity)
            zogd_capacity = np.array([x.detach().numpy() for x in zogd_capacity])
            random_search_capacity = np.array([x.detach().numpy() for x in random_search_capacity])
            SNR_results[i, 0, attempt] = simulation_capacity[-1]
            SNR_results[i, 1, attempt] = simulation_capacity_avg_with_noise[-1]
            SNR_results[i, 2, attempt] = zogd_capacity[-1]
            SNR_results[i, 3, attempt] = max(random_search_capacity)
            SNR_results[i, 4, attempt] = dnn_simulation_capacity_lst[-1]
            # SNR_results[i, 5, attempt] = zogd_capacity[zogd_time_lst-zogd_time_lst[0]< time_lst[max_index]-time_lst[0]][-1]
            # SNR_results[i, 6, attempt] = max(random_search_capacity[rand_search_time_lst - rand_search_time_lst[0] < time_lst[max_index] - time_lst[0]])
    np.save("./outputs/SNR_results.npy", SNR_results)
    plt.plot(1 / noise_array, np.mean(SNR_results[:, 0, :] / 120, axis=1))
    plt.plot(1 / noise_array, np.mean(SNR_results[:, 1, :] / 120, axis=1))
    plt.plot(1 / noise_array, np.mean(SNR_results[:, 2, :] / 120, axis=1))
    plt.plot(1 / noise_array, np.mean(SNR_results[:, 3, :] / 120, axis=1))
    plt.plot(1 / noise_array, np.mean(SNR_results[:, 4, :] / 120, axis=1))
    plt.legend(["simulation_c", "simulation with noise", "zero order gradient descent", "random search", "dnn"])
    plt.show()
def main():
    device = torch.device("cpu") # force choosing cpu since the simulation simulator works faster on CPU and it is the largest bottleneck
    print(device)
    config = Config()
    # 0. get started
    T.manual_seed(config.seed+1)  # representative results
    np.random.seed(config.seed+1)

    batch_size = config.batch_size

    # 1. create DataLoader objects

    # 2. create network
    net_diffusion = Net_diffusion(config).to(device)

    # 3. train model
    lrn_rate = 0.00005
    optim_lr = 0.002 #0.008
    load_model = config.load_model
    diffusion_train_mode = config.diffusion_train_mode
    find_optimal_lr = config.find_optimal_lr
    test_models_optimization_per_iteration = config.test_models_optimization_per_iteration
    run_snr_graph = config.run_snr_graph
    run_room_visualization = config.run_room_visualization
    show_room_graph = config.show_room_graph

    optimizer_diffusion = T.optim.Adam(net_diffusion.parameters(), lr=lrn_rate)  # weight_decay=0.001
    simulation = physfad_c(config,device=device)
    simulation.set_configuration()
    print("\nbatch_size = %3d " % batch_size)
    print("optimizer = Adam")
    print("train learning rate = %0.4f " % lrn_rate)
    print("optimization learning rate = %f " % optim_lr)


    NMSE_LST_SIZE = 10

    print("Collecting RIS configuration ")
    # generate_dataset("conditional", "", "../Data/", 32, 128, simulation, input_size=135,device=device)
    # generate_dataset("conditional", "_test", "../Data/", 32, 512, simulation, input_size=135,device=device)
    factory = physfad_factory()
    train_ds,test_ds,train_ldr, test_ldr = load_data(factory,batch_size,config,device)
    if load_model:
        print("Loading model")
        net_diffusion.load_state_dict(torch.load("./Models/Full_Main_model5.pt"))
        optimizer_diffusion = T.optim.Adam(net_diffusion.parameters(), lr=lrn_rate)

    if diffusion_train_mode:
        print("Training Diffusion Model")
        diffusion_active_training(net_diffusion,train_ldr, test_ldr, optimizer_diffusion, simulation, config, device=device)

    if find_optimal_lr:
        print("Finding Best Learning Rate for Simulation")
        find_optimal_learning_rate(simulation,test_ldr,config,device)

    if test_models_optimization_per_iteration:
        print("Testing models optimization per iteration")
        diffusion_inference(net_diffusion, test_ldr, simulation, config, device=device)

    if run_snr_graph:
        print("Running SNR graph")
        snr_graph(simulation, test_ldr, net_diffusion, config, device)

    if run_room_visualization:
        print("Generating room visualization")
        room_visualization(net_diffusion,test_ldr,simulation,config,device,optimized=False)
        room_visualization(net_diffusion,test_ldr,simulation,config,device,optimized=True)

    if show_room_graph:
        print("visualizing room graph")
        room_graph_print(simulation)

    print("\nEnd Simulation")
if __name__ == "__main__":
    enclosure = {}
    # scipy.io.loadmat("..//simulation//ComplexEnclosure.mat",enclosure)
    main()
    # cProfile.run('main()')

