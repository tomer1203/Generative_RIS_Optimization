import torch
# from memory_profiler import profile

# class rate_model_c():
#     def __init__(self,noise=None,P=None,device=torch.device("cpu")):
#         self.device = device
#         if noise == None:
#             self.noise = torch.tensor(1,device=self.device)
#         else:
#             self.noise = noise
#         self.P = P
def absolute_value_of_complex_number_64bit(complex_torch_number):
    real_part = complex_torch_number.real.type(torch.float64)
    imag_part = complex_torch_number.imag.type(torch.float64)
    return torch.sqrt(real_part**2+imag_part**2)
def capacity_loss(H, P=None, sigmaN=None, list_out=False,device=torch.device("cpu")):
    if sigmaN == None:
        sigmaN = torch.tensor(1,device=device,dtype=torch.float64)
    if P == None:
        P = torch.ones(H.shape[-3],device=device,dtype=torch.float64)
    if len(H.shape) == 4:
        H_size = H[0, 0, :, :].shape
        number_of_frequencies = H.shape[1]
        rate_freq_list = torch.zeros(len(H), number_of_frequencies)
        if torch.any(~torch.isfinite(H)):
            print("non finite found in rate model")
        _, S, _ = torch.svd(absolute_value_of_complex_number_64bit(H), some=False)
        # H_abs =
        S = S.to(H.device).type(torch.float64)
        S_N = torch.zeros([S.shape[0], S.shape[1], S.shape[2], S.shape[2]], device=H.device, dtype=torch.float64)
        S_N.diagonal(dim1=-2, dim2=-1).copy_(1 + S * S * P.reshape(1, -1, 1) / sigmaN)
        rate_freq_list = torch.log2(torch.det(S_N))
        # for i in range(len(H)): # inside the batch
        #     for f in range(number_of_frequencies):
        #         Sf = torch.squeeze(S[i,f,:])
        #         rate_freq_list[i,f] = torch.log2(torch.prod(1 + torch.norm(Sf)**2 * P[f] / sigmaN))
        if list_out:
            return torch.sum(rate_freq_list, dim=1)
        return torch.sum(rate_freq_list) / H.shape[0]
    else:
        H_size = H[1, :, :].shape
        number_of_frequencies = len(H)
        rate_freq_list = torch.zeros(number_of_frequencies,dtype=torch.float64)
        for f in range(number_of_frequencies):
            Hf = torch.squeeze(H[f, :, :])
            if torch.any(~torch.isfinite(Hf)):
                print("non finite found in rate model er:2")
            _, S, _ = torch.svd(absolute_value_of_complex_number_64bit(Hf), some=True)
            rate_freq_list[f] = torch.log2(torch.det(torch.diag(1 + S * S * P[f] / sigmaN)))
        return torch.sum(rate_freq_list)