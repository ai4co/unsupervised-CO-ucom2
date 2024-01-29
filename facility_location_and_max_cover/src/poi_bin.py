import torch

def pmf_poibin(prob_matrix_raw, device, use_normalization=True):
    if use_normalization:                
        prob_matrix = torch.sigmoid(prob_matrix_raw)
    else:
        prob_matrix = torch.zeros_like(prob_matrix_raw)
        prob_matrix += prob_matrix_raw
    number_vec = prob_matrix.shape[0]
    number_trials = prob_matrix.shape[1]
    omega = torch.tensor(2 * torch.pi / (number_trials + 1), dtype=torch.float, device=device)
    chi = torch.empty(number_vec, number_trials + 1, device=device, dtype=torch.cfloat)
    chi[:, 0] = 1.0
    half_number_trials = int(number_trials / 2 + number_trials % 2)
    exp_value = torch.exp(omega * torch.arange(1, half_number_trials + 1, device=device) * 1j)
    xy = (1.0 - prob_matrix.unsqueeze(1) + prob_matrix.unsqueeze(1) * exp_value[:, None])
    argz_sum = torch.arctan2(xy.imag, xy.real).sum(dim=-1)
    exparg = torch.log(torch.abs(xy)).sum(dim=-1)
    d_value = torch.exp(exparg)
    chi[:, 1 : half_number_trials + 1] = d_value * torch.exp(argz_sum * 1j)
    chi[:, half_number_trials + 1 : number_trials + 1] = torch.conj(
            chi[:, 1 : number_trials - half_number_trials + 1].flip(1))
    chi /= number_trials + 1
    xi = torch.fft.fft(chi)        
    xi = xi.real.float()        
    xi += torch.finfo(xi.dtype).eps
    return xi

def pmf_poibin_vec(prob_vector_raw, device, use_normalization=True):
    if use_normalization:                
        prob_vector = torch.sigmoid(prob_vector_raw)
    else:
        prob_vector = torch.zeros_like(prob_vector_raw)
        prob_vector += prob_vector_raw
    number_trials = prob_vector.shape[0]
    omega = torch.tensor(2 * torch.pi / (number_trials + 1), dtype=torch.float, device=device)
    chi = torch.empty(number_trials + 1, device=device, dtype=torch.cfloat)
    chi[0] = 1.0
    half_number_trials = int(number_trials / 2 + number_trials % 2)
    exp_value = torch.exp(omega * torch.arange(1, half_number_trials + 1, device=device) * 1j)
    xy = (1.0 - prob_vector + prob_vector * exp_value[:, None])
    argz_sum = torch.arctan2(xy.imag, xy.real).sum(dim=-1)
    exparg = torch.log(torch.abs(xy)).sum(dim=1)
    d_value = torch.exp(exparg)
    chi[1 : half_number_trials + 1] = d_value * torch.exp(argz_sum * 1j)
    chi[half_number_trials + 1 : number_trials + 1] = torch.conj(
            chi[1 : number_trials - half_number_trials + 1].flip(0))
    chi /= number_trials + 1
    xi = torch.fft.fft(chi)        
    xi = xi.real.float()        
    xi += torch.finfo(xi.dtype).eps
    return xi