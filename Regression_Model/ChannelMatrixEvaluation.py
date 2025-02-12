import scipy.io
import numpy as np
import torch
import random
# from PhysFadPy import GetH,get_bessel_w
import matplotlib.pyplot as plt
import cProfile
import cProfile,pstats,io
import datetime
from rate_model import capacity_loss







# def test_configurations_capacity(parameters,ris_configuration,W,device,list_out=False,noise=1):
#     H = physfad_model(parameters, ris_configuration,W, device)
#     if len(H.shape)==4:
#         return capacity_loss(H,torch.ones(H.shape[1],device=device),noise,list_out=list_out),H
#     return capacity_loss(H,torch.ones(H.shape[0]),noise,list_out=list_out),H

def test_configurations_capacity(physfad,ris_configuration,tx_x,tx_y,device,list_out=False,noise=None):
    tx_size = tx_x.shape[0]
    ris_configuration_size = ris_configuration.shape[0]
    batch_size = ris_configuration_size // tx_size
    if tx_size != 1:
        H = torch.zeros([ris_configuration_size, physfad.config.output_size,physfad.config.output_shape[0],physfad.config.output_shape[1]],dtype=torch.complex64)
        for i in range(len(tx_x)):
            batch_of_H = physfad(ris_configuration[i * batch_size:(i + 1) * batch_size], tx_x[i].unsqueeze(0), tx_y[i].unsqueeze(0))
            if batch_size == 1:
                batch_of_H = batch_of_H.unsqueeze(0)
            H[i*batch_size:(i+1)*batch_size] = batch_of_H
        return capacity_loss(H, sigmaN=noise, list_out=list_out, device=device), H
    H = physfad(ris_configuration,tx_x,tx_y)
    return capacity_loss(H,sigmaN=noise,list_out=list_out,device=device),H


def physfad_channel_optimization(device,physfad,starting_inp=None,tx_x=None,tx_y=None,noise_power = 1, learning_rate=0.005,num_of_iterations=150):
    iters = 0
    # num_of_iterations = 150

    if starting_inp is None:
        estOptInp = torch.randn([1, physfad.N_RIS_PARAMS], requires_grad=True, device=device)
        # tx_x = torch.randn([1,3],requires_grad=False,device=device)
        # tx_y = torch.randn([1,3],requires_grad=False,device=device)
        batch_size = 1
    else:
        estOptInp = starting_inp
        batch_size = starting_inp.shape[0]
    # old_result = torch.from_numpy(np.loadtxt("Physfad_optimal_parameters.txt"))
    # estOptInp = old_result.unsqueeze(0).clone().detach().to(device).requires_grad_(True)
    # scipy.io.savemat("estOptInp.mat", {"estOptInp": estOptInp.cpu().detach().numpy()})

    time_lst = []
    physfad_capacity_lst = []
    Inp_optimizer = torch.optim.SGD([estOptInp], lr=learning_rate) # 0.1
    current_loss = torch.Tensor([1])
    # H = torch.zeros([batch_size,physfad.config.output_size,physfad.config.output_shape[0],physfad.config.output_shape[1]],dtype=torch.complex64)
    while (current_loss.item() > -600 and iters < num_of_iterations):
        Inp_optimizer.zero_grad()
        estOptInp_norm = torch.nn.functional.sigmoid(estOptInp)
        # estOptInp_norm = estOptInp
        # for b in range(batch_size):
        #     H[b] = physfad(estOptInp_norm[b].unsqueeze(0),tx_x[b].unsqueeze(0),tx_y[b].unsqueeze(0))
        H = physfad(estOptInp_norm,tx_x,tx_y)
        # scipy.io.savemat("H_python_mat.mat", {"H_python_mat": H.cpu().detach().numpy()})
        # loss = -torch.sum(torch.abs(H[:,0,1]))
        loss = -capacity_loss(H, sigmaN = noise_power,device=device)
        time_lst.append(datetime.datetime.now())
        physfad_capacity_lst.append(-loss.item())
        loss.backward()
        Inp_optimizer.step()
        # def closure():
        #     optimizer.zero_grad()
        #     pred = net(estOptInp)
        #     loss = -torch.sum(pred)
        #     # grad = torch.autograd.grad(loss,net.parameters(),create_graph=True)
        #     loss.backward()
        #     return loss
        # Inp_optimizer.step(closure)

        if iters % 1 == 0:
            print("physfad iteration #{0} input distance from zero: {1} loss: {2}".format(iters, torch.abs(
                estOptInp_norm).sum().cpu().item(), -loss.item()))
            # if loss.item() < -100:
            #     plt.plot(torch.abs(H[:, 0, 1]).cpu().detach().numpy())
            #     plt.show()

        iters = iters + 1
    return time_lst,physfad_capacity_lst,estOptInp_norm
def random_search_optimization(physfad,iteration_limit=300,device='cpu',time_limit=None,noise_power=1, initial_inp = None,tx_x=None,tx_y=None):
    inp_size = 264
    iters = 0
    # num_of_iterations = 200
    assert tx_x is not None, "You might want to set tx location"
    assert tx_y is not None, "You might want to set tx location"

    time_lst = []
    random_search_capacity_lst = []
    # Inp_optimizer = torch.optim.Adam([estOptInp], lr=0.01)
    current_loss = 1
    from utils import zo_estimate_gradient
    capacity_physfad = lambda x : -capacity_loss(physfad(x,tx_x,tx_y), sigmaN=noise_power)
    while ((time_limit is None or time_lst[-1]-time_lst[0]<time_limit) and iters < iteration_limit):
        if iters == 0 and initial_inp != None:
            estOptInp = initial_inp
        else:
            estOptInp = torch.randn([1, inp_size], device=device).cpu().detach().numpy()
        out = capacity_physfad(estOptInp)
        time_lst.append(datetime.datetime.now())
        random_search_capacity_lst.append(-out)
        # def closure():
        #     optimizer.zero_grad()
        #     pred = net(estOptInp)
        #     loss = -torch.sum(pred)
        #     # grad = torch.autograd.grad(loss,net.parameters(),create_graph=True)
        #     loss.backward()
        #     return loss
        # Inp_optimizer.step(closure)

        if iters % 1 == 0:
            print("rand iteration #{0} input distance from zero: {1} loss: {2}".format(iters, np.abs(
                estOptInp).sum(), -out))

        iters = iters + 1
    return time_lst, random_search_capacity_lst
def zeroth_grad_optimization(device,physfad,starting_inp=None,tx_x=None,tx_y=None,noise_power=1,num_of_iterations=200):
    inp_size = 264
    iters = 0
    # num_of_iterations = 200
    assert tx_x is not None,"You might want to set tx location"
    assert tx_y is not None,"You might want to set tx location"
    if starting_inp is None:
        estOptInp = torch.rand([1, inp_size], requires_grad=True, device=device).cpu().detach().numpy()
        # tx_x = torch.randn([1,3],requires_grad=False,device=device).cpu().detach().numpy()
        # tx_y = torch.randn([1, 3], requires_grad=False, device=device).cpu().detach().numpy()
    else:
        estOptInp = starting_inp

    epsilon = 0.0001
    m = 4
    lr = 0.1
    time_lst = []
    physfad_capacity_lst = []
    gradient_score_lst = []
    # Inp_optimizer = torch.optim.Adam([estOptInp], lr=0.01)
    current_loss = 1
    from utils import zo_estimate_gradient,cosine_score
    capacity_physfad = lambda x,tx_x_arg,tx_y_arg : -capacity_loss(physfad(x,tx_x_arg,tx_y_arg), sigmaN=noise_power,device=device)
    while (current_loss > -300 and iters < num_of_iterations):
        # estOptInp.grad = None

        grad_inp = zo_estimate_gradient(capacity_physfad, estOptInp, tx_x, tx_y, epsilon, m, device)
        # physfad_capacity, physfad_H = test_configurations_capacity(physfad, estOptInp, tx_x, tx_y, device, list_out=False, noise=noise_power)
        # physfad_grad = -torch.autograd.grad(physfad_capacity, estOptInp, retain_graph=True)[0]
        # gradient_score = cosine_score(physfad_grad, grad_inp)
        # print(gradient_score)
        # gradient_score_lst.append(gradient_score)
        estOptInp = estOptInp - lr * grad_inp
        estOptInp = torch.clip(estOptInp,0,1)
        out = capacity_physfad(estOptInp, tx_x, tx_y)
        time_lst.append(datetime.datetime.now())
        physfad_capacity_lst.append(-out)
        # def closure():
        #     optimizer.zero_grad()
        #     pred = net(estOptInp)
        #     loss = -torch.sum(pred)
        #     # grad = torch.autograd.grad(loss,net.parameters(),create_graph=True)
        #     loss.backward()
        #     return loss
        # Inp_optimizer.step(closure)

        if iters % 1 == 0:
            print("zogd iteration #{0} input distance from zero: {1} loss: {2}".format(iters, torch.abs(estOptInp).sum().cpu().item(), -out))
            # print(gradient_score)
        iters = iters + 1
    return time_lst, physfad_capacity_lst,gradient_score_lst
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(device)
    zeroth_grad_optimization(device)
    # func = lambda x:(x**2).sum()
    # from utils import estimate_gradient
    # print()
    # x = np.array([1,0])
    # lr = 0.2
    # for i in range(100):
    #     print(x)
    #     grad_x = estimate_gradient(func,x, 0.0000001, 1000)
    #     x = x-lr*grad_x
    # physfad_channel_optimization(device)
    print("Done optimization")

    # print(estOptInp)
    # np.savetxt("Physfad_optimal_parameters.txt", estOptInp[0,:].cpu().detach().numpy())

    # plt.plot(torch.mean(test_pred, dim=0).t().cpu().detach().numpy())
    # plt.legend(["asa,asd"])
    # plt.show()
    # plt.plot(torch.abs(H[:,0,1]))
    # plt.show()
if __name__ == "__main__":
    # cProfile.run('main()')
    main()