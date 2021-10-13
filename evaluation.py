import torch
from tools.compute_metrics_norm import compute_metrics_norm, pesq2
from utils import *
from joblib import Parallel, delayed
import scipy
import logging
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
import numpy as np
from natsort import natsorted


class Eval:
    def __init__(self, model, test_data, threshold, batch_size):
        self.model = model
        self.threshold = threshold
        self.test_data = test_data
        self.batch_size = batch_size

    def test_step(self, batch, metrics_list, wav_name):
        clean_stft = batch[0].cuda()
        # clean_stft = clean_stft / torch.max(torch.abs(clean_stft))
        noisy_stft = batch[1].cuda()
        clean_audio = batch[2].cuda()
        noisy_audio = batch[3].cuda()
        clean_stft_mag = batch[4].cuda()
        noisy_stft_mag = batch[5].cuda()
        out_mag, out_complex, att_mask = self.model(noisy_stft, noisy_stft_mag)
        ISTFT_loss, val_output, val_target = \
            calculate_loss(out_mag, clean_audio, out_complex, self.threshold, minmax=False)
        # print(wav_name[4])
        #
        # att_mask_1_1 = att_mask[0].squeeze()
        # att_mask_1_2 = att_mask[1].squeeze()
        # att_mask_1_3 = att_mask[2].squeeze()
        # plt.imshow(np.flip(att_mask_1_1[4].cpu().numpy(), 0))
        # plt.show()
        # plt.imshow(np.flip(att_mask_1_2[4].cpu().numpy(), 0))
        # plt.show()
        # plt.imshow(np.flip(att_mask_1_3[4].cpu().numpy(), 0))
        # plt.show()
        # plt.plot(noisy_audio[4].cpu().numpy())
        # plt.show()
        # plt.plot(val_output[4].numpy())
        # plt.show()
        # plt.plot(val_target[4].numpy())
        # plt.show()
        # noisy_phase = torch.atan2(noisy_stft[:,1,:,:], noisy_stft[:,0,:,:])
        # plt.imshow(np.flip(noisy_phase.squeeze(1)[1].cpu().numpy(), 0))
        # plt.show()
        # est_phase = torch.atan2(out_complex[:, 1, :, :], out_complex[:, 0, :, :])
        # plt.imshow(np.flip(est_phase.squeeze(1)[1].cpu().numpy(), 0))
        # plt.show()
        # clean_phase = torch.atan2(clean_stft[:, 1, :, :], clean_stft[:, 0, :, :])
        # plt.imshow(np.flip(clean_phase.squeeze(1)[1].cpu().numpy(), 0))
        # plt.show()
        # plt.imshow(np.flip(noisy_stft_mag.squeeze()[1].cpu().numpy(), 0))
        # plt.show()
        # plt.imshow(np.flip(out_mag.squeeze()[1].cpu().numpy(), 0))
        # plt.show()
        # plt.imshow(np.flip(clean_stft_mag.squeeze()[1].cpu().numpy(), 0))
        # plt.show()

        metrics_score = Parallel(n_jobs=1)(delayed(metrics)(c, n, name) for c, n, name in zip(val_target, val_output, wav_name))
        # metrics_score = metrics(val_target, val_output, wav_name)
        metrics_score = np.squeeze(np.array(metrics_score))
        metrics_list_val = np.sum(metrics_score, axis=0)
        metrics_list += metrics_list_val

        return metrics_list, val_output, val_target

    def test(self):
        metrics_list = np.zeros(6)
        clean_audio_path = r'/no_backups/s1374/Demand_16kHz/test/clean'
        clean_audio_set = os.listdir(clean_audio_path)
        clean_audio_set = natsorted(clean_audio_set)
        clean_idx = 0
        metrics_list_all_avg = np.zeros(6)
        # cat_audio = []
        # cl_cat_audio = []
        unit_idx = 0
        with torch.no_grad():
            for index, [batch, length, wav_name] in enumerate(self.test_data):
                step_test = index + 1
                test_output, est_sig, cl_sig = self.test_step(batch, metrics_list, wav_name)
                est_sig = est_sig[0].numpy()
                cl_sig = cl_sig[0].numpy()
                sr, clean_audio = read(os.path.join(clean_audio_path, clean_audio_set[clean_idx]))
                len_audio = len(clean_audio)
                units = len_audio//16384
                # calculate overall track metric
                if wav_name[0][:8] == clean_audio_set[clean_idx][:8] and unit_idx == 0:
                    cat_audio = est_sig
                    cl_cat_audio = cl_sig
                    unit_idx += 1
                elif wav_name[0][:8] == clean_audio_set[clean_idx][:8] and 0 < unit_idx < units:
                    cat_audio = np.concatenate((cat_audio, est_sig))
                    cl_cat_audio = np.concatenate((cl_cat_audio, cl_sig))
                    unit_idx += 1
                elif wav_name[0][:8] == clean_audio_set[clean_idx][:8] and unit_idx == units:
                    cat_audio = np.concatenate((cat_audio, est_sig[-(len_audio % 16384):]))
                    cl_cat_audio = np.concatenate((cl_cat_audio, cl_sig[-(len_audio % 16384):]))
                    # cat_audio = normalize_signal(cat_audio, False)
                    # cl_cat_audio = normalize_signal(cl_cat_audio, False)
                    assert len(cat_audio) == len_audio
                    metrics_list_all = compute_metrics_norm(cl_cat_audio, cat_audio, 16000, 0, 0, 0)
                    metrics_list_all = np.array(metrics_list_all)
                    metrics_list_all_avg += metrics_list_all
                    unit_idx = 0
                    clean_idx += 1


        metrics_list = test_output / (step_test * self.batch_size)
        mean_pesq = metrics_list[0]
        mean_csig = metrics_list[1]
        mean_cbak = metrics_list[2]
        mean_covl = metrics_list[3]
        mean_ssnr = metrics_list[4]
        mean_stoi = metrics_list[5]
        metrics_list_all_avg = metrics_list_all_avg / clean_idx

        template = '\n pesq: {}, csig: {}, ' \
                   'cbak: {}, covl: {}, ssnr: {}, stoi: {} \n metrics_all: {}, num: {}'
        logging.info(
            template.format(mean_pesq, mean_csig, mean_cbak, mean_covl, mean_ssnr, mean_stoi, metrics_list_all_avg, clean_idx))


def metrics(clean, noisy, wav_name):
    clean = clean.numpy()
    noisy = noisy.numpy()
    # path = os.path.join("/usrhomes/s1374/PycharmProjects/Attention_Unet_PlusPlus_phase/est_audio_BO", wav_name)
    # scipy.io.wavfile.write(path, 16000, noisy)
    mean_pesq_output_val_sliced, mean_csig_output_val_sliced, \
    mean_cbak_output_val_sliced, mean_covl_output_val_sliced, \
    mean_ssnr_output_val_sliced, mean_stoi_output_val_sliced = compute_metrics_norm(
        clean, noisy, 16000, 0, 0, 0)
    return mean_pesq_output_val_sliced, mean_csig_output_val_sliced, mean_cbak_output_val_sliced, \
           mean_covl_output_val_sliced, mean_ssnr_output_val_sliced, mean_stoi_output_val_sliced


def normalize_signal(vec, min_max=False):
    max_vec = np.max(vec)
    min_vec = np.min(vec)
    if min_max:
        vec = (2*vec - max_vec - min_vec)/(max_vec - min_vec)
        # vec = (vec - min_vec) / (max_vec - min_vec)
    else:
        # mean = np.mean(vec)
        # std = np.std(vec)
        # vec = (vec - mean) / std
        vec = vec / np.max(np.abs(vec))

    return vec