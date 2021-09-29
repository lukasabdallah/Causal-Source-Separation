import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import os


def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.ConvTranspose2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.InstanceNorm2d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif type(m) == nn.BatchNorm2d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def seed_torch(seed=20):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def scale_mag_new(s, a, b):
    sRescale = (((b - a) * (s - torch.min(s))) / (torch.max(s) - torch.min(s) + 1e-12)) + a
    return sRescale


def scale_mag(s, size, threshold=-95):
    a_1 = torch.full(size, threshold, dtype=torch.float)
    rescale = torch.where(torch.lt(s, threshold), a_1, s)
    return rescale


def normalise_spect(s):
    return torch.div(torch.sub(s, torch.min(s)), (torch.sub(torch.max(s), torch.min(s)) + 1e-12))


def conv_complex_after_stft(a, b, complex=True):
    first_term = a * torch.cos(b)
    second_term = a * torch.sin(b)
    if not complex:
        return torch.stack([first_term, second_term], 0)
    else:
        return torch.complex(first_term, second_term)


def real2complex(a, b):
    first_term = a * torch.cos(b)
    second_term = a * torch.sin(b)
    out = torch.stack([first_term, second_term], -1)
    return out


def normalize_tensor(s, minmax=False):
    if not minmax:
        # s = s - torch.mean(s)
        vec = torch.div(s, torch.max(torch.abs(s)))
    else:
        vec = (2*s-torch.min(s)-torch.max(s)) / (torch.max(s) - torch.min(s) + 1e-12)
    return vec


def normalize_audio(s):
    vec = (2 * s - torch.min(s) - torch.max(s)) / (torch.max(s) - torch.min(s) + 1e-12)
    return vec


def preprocess(image):
    return image * 2 - 1


def stft(x, n_fft, hop_length, win_length, window, center, onesided, return_complex):
    if window == 'hann':
        window = torch.hann_window(win_length)
    elif window == 'hamm':
        window = torch.hamming_window(win_length)
    out = torch.stft(input=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
    window=window, normalized=True, center=center, onesided=onesided, return_complex=return_complex)
    return out


def istft(x, n_fft, hop_length, win_length, window, length=None, center=True, onesided=True, return_complex=False):
    if window == 'hann':
        window = torch.hann_window(win_length).cuda()
    elif window == 'hamm':
        window = torch.hamming_window(win_length).cuda()
    out = torch.istft(input=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
    window=window, center=center, normalized=True, onesided=onesided, length=length, return_complex=return_complex)

    return out


def calculate_loss(outputs, target_to_check, input_phase, threshold, minmax=False):
    ISTFT_loss_final = 0.0
    batch_size = outputs.size()[0]
    validation_output = []
    validation_target = []
    output_for_ISTFT = scale_mag_new(outputs, threshold, 0)
    output_for_ISTFT = torch.pow(10, torch.div(output_for_ISTFT, 20))
    input_phase = input_phase.detach()
    input_phase_to_save = torch.atan2(input_phase[:,1,:,:], input_phase[:,0,:,:])
    new_signal_for_istft = conv_complex_after_stft(output_for_ISTFT.squeeze(1), input_phase_to_save.squeeze(1))
    for i in range(batch_size):
        new_signal_after_istft_final = istft(new_signal_for_istft[i], 511, 64, 511, 'hamm', length=16384).squeeze()
        signal_to_check_final = target_to_check[i]
        new_signal_after_istft_final = normalize_tensor(new_signal_after_istft_final, minmax)
        signal_to_check_final = normalize_tensor(signal_to_check_final, minmax)
        ISTFT_loss = F.mse_loss(new_signal_after_istft_final, signal_to_check_final)
        ISTFT_loss_final = ISTFT_loss_final + torch.log(ISTFT_loss)
        assert len(new_signal_after_istft_final) == len(signal_to_check_final)
        validation_output.append(new_signal_after_istft_final.detach().cpu())
        validation_target.append(signal_to_check_final.detach().cpu())

    return ISTFT_loss_final / batch_size, validation_output, validation_target
