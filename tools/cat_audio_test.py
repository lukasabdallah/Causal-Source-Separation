import os
from natsort import natsorted
import torchaudio
import numpy as np
from tools.compute_metrics_norm import compute_metrics_norm
from scipy.io.wavfile import read, write


def normalize_audio(s, minmax=True):
    if not minmax:
        vec = s - np.mean(s)
    else:
        vec = (2*s - np.min(s) - np.max(s)) / (np.max(s) - np.min(s))
    return vec


est_path = "/usrhomes/s1374/PycharmProjects/Attention_Unet_PlusPlus_phase/est_audio_BO"
clean_path = "/no_backups/s1374/Demand_16kHz/clean_testset_16kHz"
clean_path_cut = "/no_backups/s1374/dataset16kHz_cut_16384/clean_test_all"

est_wav_name = os.listdir(est_path)
est_wav_name = natsorted(est_wav_name)
clean_wav_name = os.listdir(clean_path)
clean_wav_name = natsorted(clean_wav_name)
clean_cut_name = os.listdir(clean_path_cut)
clean_cut_name = natsorted(clean_cut_name)

start = True
metrics_list_total = np.zeros(6)
step = 0

for name in clean_wav_name:
    clean_wav_path = os.path.join(clean_path, name)
    _, clean_sig = read(clean_wav_path)
    len_clean = len(clean_sig)
    units = len_clean//16384
    est_audio = []
    cl_audio = []
    for i in range(units):
        est_audio_path = os.path.join(est_path, name[:-4] + '_' + str(i) + '.wav')
        cl_audio_path = os.path.join(clean_path_cut, name[:-4] + '_' + str(i) + '.wav')
        _, est_sig = read(est_audio_path)
        _, cl_sig = read(cl_audio_path)
        cl_sig = cl_sig / np.max(np.abs(cl_sig))
        cl_sig = normalize_audio(cl_sig)
        est_sig = normalize_audio(est_sig)
        est_audio.append(est_sig)
        cl_audio.append(cl_sig)
    if len_clean % 16384 != 0:
        est_audio_path = os.path.join(est_path, name[:-4] + '_' + str(units) + '.wav')
        cl_audio_path = os.path.join(clean_path_cut, name[:-4] + '_' + str(units) + '.wav')
        _, est_sig = read(est_audio_path)
        _, cl_sig = read(cl_audio_path)
        cl_sig = cl_sig / np.max(np.abs(cl_sig))
        cl_sig = normalize_audio(cl_sig)
        est_sig = normalize_audio(est_sig)
        est_audio.append(est_sig[-(len_clean % 16384):])
        cl_audio.append(cl_sig[-(len_clean % 16384):])
    est_track = np.concatenate(est_audio, 0)
    cl_track = np.concatenate(cl_audio, 0)
    assert len(est_track) == len_clean
    # path = os.path.join("/usrhomes/s1374/PycharmProjects/Attention_Unet_PlusPlus_phase/est_audio_cat_BO", name)
    # write(path, 16000, est_track)
    # cl_track = normalize_audio(cl_track, minmax=True)
    # est_track = normalize_audio(est_track, minmax=True)
    metric_list = compute_metrics_norm(cl_track, est_track, 16000, 0, 0, 0)
    metrics_list_total += np.array(metric_list)
    step += 1

metrics_list_mean = metrics_list_total / step
print(metrics_list_mean, step)

# for name in est_wav_name:
#     est_wav_path = os.path.join(est_path, name)
#     clean_wav_path = os.path.join(clean_path, name)
#     _, est_sig = read(est_wav_path)
#     _, clean_sig = read(clean_wav_path)
#     len_clean = len(clean_sig)
#
#     if eval(name[-5]) == 0 and start:
#         est_audio = [est_sig]
#         n_last = 0
#     elif eval(name[-5]) > 0:
#         n_last = eval(name[-5])
#         est_audio.append(est_sig)
#     elif eval(name[-5]) == 0 and not start:
#         flag = 0
#         est_track = np.concatenate(est_audio, 0)
#         print(np.shape(est_track))
#         clean_track = normalize_audio(clean_track, minmax=True)
#         est_track = normalize_audio(est_track, minmax=True)
#         metric_list = compute_metrics_norm(clean_track, est_track, 16000, 0, 0, 0)
#         metrics_list_total += np.array(metric_list)
#         step += 1
#         clean_audio = [clean_sig]
#         est_audio = [est_sig]
#     start = False
#
# clean_track = np.concatenate(clean_audio, 0)
# est_track = np.concatenate(est_audio, 0)
# clean_track = normalize_audio(clean_track, minmax=True)
# est_track = normalize_audio(est_track, minmax=True)
# metric_list = compute_metrics_norm(clean_track, est_track, 16000, 0, 0, 0)
# metrics_list_total += np.array(metric_list)
# step += 1
#
# metrics_list_mean = metrics_list_total / step
# print(metrics_list_mean, step)