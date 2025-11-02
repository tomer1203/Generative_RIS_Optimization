import torch
import tensorflow as tf

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

def absolute_value_of_complex_number_64bit_tf(complex_tf_tensor):
    real_part = tf.cast(tf.math.real(complex_tf_tensor), tf.float64)
    imag_part = tf.cast(tf.math.imag(complex_tf_tensor), tf.float64)
    return tf.sqrt(real_part**2 + imag_part**2)
def capacity_loss_tf(H, P=None, sigmaN=None, list_out=False):
    if sigmaN is None:
        sigmaN = tf.constant(1.0, dtype=tf.float64)
    if P is None:
        P = tf.ones(shape=(H.shape[-3],), dtype=tf.float64)

    if len(H.shape) == 4:
        H_size = H[0, 0, :, :].shape
        number_of_frequencies = H.shape[1]
        rate_freq_list = tf.zeros((H.shape[0], number_of_frequencies), dtype=tf.float64)

        # if not tf.reduce_all(tf.math.is_finite(H)):
        #     tf.print("non finite found in rate model")

        # Apply absolute value function on complex input
        H_abs = absolute_value_of_complex_number_64bit_tf(H)
        S = tf.linalg.svd(H_abs, compute_uv=False)
        S = tf.cast(S, tf.float64)

        # Construct diagonal matrices with 1 + S^2 * P / sigmaN
        P_reshaped = tf.reshape(P, (1, -1, 1))
        S_squared = S * S
        diag_vals = 1.0 + S_squared * P_reshaped / sigmaN

        # Build diagonal matrices and take determinant
        eye_shape = tf.shape(S)
        def make_diag_matrix(diag):
            return tf.linalg.set_diag(tf.zeros((eye_shape[-1], eye_shape[-1]), dtype=tf.float64), diag)

        diag_matrices = tf.map_fn(lambda x: tf.linalg.diag(x), diag_vals, dtype=tf.float64)
        det_vals = tf.linalg.det(diag_matrices)
        rate_freq_list = tf.math.log(det_vals) / tf.math.log(tf.constant(2.0, dtype=tf.float64))

        if list_out:
            return tf.reduce_sum(rate_freq_list, axis=1)
        return tf.reduce_sum(rate_freq_list) / tf.cast(tf.shape(H)[0], tf.float64)

    else:
        H_size = H[1, :, :].shape
        number_of_frequencies = H.shape[0]
        rate_freq_list = tf.zeros((number_of_frequencies,), dtype=tf.float64)

        for f in range(number_of_frequencies):
            Hf = tf.squeeze(H[f, :, :])
            # if not tf.reduce_all(tf.math.is_finite(Hf)):
            #     tf.print("non finite found in rate model er:2")

            S = tf.linalg.svd(absolute_value_of_complex_number_64bit_tf(Hf), compute_uv=False)
            S = tf.cast(S, tf.float64)

            diag_vals = 1.0 + S * S * P[f] / sigmaN
            diag_matrix = tf.linalg.diag(diag_vals)
            rate_freq_list = tf.tensor_scatter_nd_update(rate_freq_list, [[f]],
                [tf.math.log(tf.linalg.det(diag_matrix)) / tf.math.log(tf.constant(2.0, dtype=tf.float64))])

        return tf.reduce_sum(rate_freq_list)

def convert_to_torch(tf_tensor):
    (batch_size,num_rx,num_rx_ant,num_tx,num_tx_ant,time_steps,frequencies) = tf_tensor.shape
    tf_reshaped = tf.reshape(tf_tensor,(batch_size,num_rx*num_rx_ant,num_tx*num_tx_ant,frequencies)) # combine antenna dims
    tf_reshaped = tf.transpose(tf_reshaped,perm=[0,3,1,2]) # frequency before tx-rx
    torch_tensor = torch.from_numpy(tf_reshaped.numpy())
    return torch_tensor
def convert_shape(tf_tensor):
    (batch_size,num_rx,num_rx_ant,num_tx,num_tx_ant,time_steps,frequencies) = tf_tensor.shape
    tf_reshaped = tf.reshape(tf_tensor,(batch_size,num_rx*num_rx_ant,num_tx*num_tx_ant,frequencies)) # combine antenna dims
    tf_reshaped = tf.transpose(tf_reshaped,perm=[0,3,1,2]) # frequency before tx-rx
    return tf_reshaped