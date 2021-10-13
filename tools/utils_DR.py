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
    elif type(m) == nn.GroupNorm:
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


def overlap_calculation(wav_length):
    overlap = (1023 - np.ceil((wav_length / 512))).astype(int)
    return overlap


def adapt_length(target_audio, wav_length, length_for_stft):
    if length_for_stft >= wav_length:
        target_audio = np.concatenate((target_audio, target_audio[0:length_for_stft-wav_length]), axis=-1)
    else:
        target_audio = target_audio[0:length_for_stft]
    return target_audio


def adapt_length_torch(target_audio, wav_length, length_for_stft):
    if length_for_stft >= wav_length:
        target_audio = torch.cat((target_audio, target_audio[0:length_for_stft-wav_length]), dim=-1)
    else:
        target_audio = target_audio[0:length_for_stft]
    return target_audio


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


def calculate_DR_loss(outputs, overlap_for_stft, target_to_check, input_phase, audio_length, threshold):
    ISTFT_loss_final = 0.0
    batch_size = outputs.size()[0]
    validation_output = []
    validation_target = []
    extra_length = []
    actual_length = []
    validation_output_length = target_to_check[0].size()[0]
    output_for_ISTFT = scale_mag_new(outputs, threshold, 0)
    output_for_ISTFT = torch.pow(10, torch.div(output_for_ISTFT, 20))
    input_phase = input_phase.detach()
    input_phase_to_save = torch.atan2(input_phase[:,1,:,:], input_phase[:,0,:,:])
    new_signal_for_istft = conv_complex_after_stft(output_for_ISTFT.squeeze(1), input_phase_to_save.squeeze(1))

    for i in range(batch_size):
        new_signal_after_istft = istft(new_signal_for_istft[i], 1023, overlap_for_stft[i], 1023, 'hamm', center=False,
                                       length=audio_length[i])
        target_to_check_after_istft = target_to_check[i, 0: audio_length[i]].float()

        new_signal_after_istft_final = normalize_tensor(new_signal_after_istft)
        signal_to_check_final = normalize_tensor(target_to_check_after_istft)
        ISTFT_loss = F.mse_loss(new_signal_after_istft_final, signal_to_check_final)

        ISTFT_loss_final = ISTFT_loss_final + torch.log(ISTFT_loss)
        extra_length_individual = validation_output_length - new_signal_after_istft_final.size()[0]
        actual_length_individual = new_signal_after_istft_final.size()[0]

        assert len(new_signal_after_istft_final) == len(signal_to_check_final)

        validation_output.append(new_signal_after_istft_final.detach().cpu())
        validation_target.append(signal_to_check_final.detach().cpu())
        extra_length.append(extra_length_individual)
        actual_length.append(actual_length_individual)

    return ISTFT_loss_final / batch_size, validation_output, validation_target, extra_length, actual_length