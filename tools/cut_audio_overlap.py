import os
from scipy.io import wavfile


train_noisy_ds = os.listdir(r'/no_backups/s1374/Demand_16kHz/noisy_trainset_28spk_16kHz')
train_clean_ds = os.listdir(r'/no_backups/s1374/Demand_16kHz/clean_trainset_28spk_16kHz')

for noisy_sample in train_noisy_ds:
    test_dir = os.path.join(r'/no_backups/s1374/Demand_16kHz/noisy_trainset_28spk_16kHz', noisy_sample)
    samplerate, data = wavfile.read(test_dir)
    assert samplerate == 16000
    length_train = len(data)
    cut_unit = int(length_train/(16384//2)) - 1
    sample_list = []
    for i in range(cut_unit):
        j = i*0.5
        sample = data[int(j*16384): int((j+1)*16384)]
        sample_list.append(sample)
    if length_train % (16384//2) != 0:
        sample = data[-16384:]
        sample_list.append(sample)
    for j, audio in enumerate(sample_list):
        write_path = os.path.join(r'/no_backups/s1374/dataset16kHz_16384_overlap/noisy_train', noisy_sample[:-4] + '_' + str(j) + '.wav')
        wavfile.write(write_path, samplerate, audio)

for clean_sample in train_clean_ds:
    test_dir = os.path.join(r'/no_backups/s1374/Demand_16kHz/clean_trainset_28spk_16kHz', clean_sample)
    samplerate, data = wavfile.read(test_dir)
    assert samplerate == 16000
    length_train = len(data)
    cut_unit = int(length_train/(16384//2)) - 1
    sample_list = []
    for i in range(cut_unit):
        j = i*0.5
        sample = data[int(j*16384): int((j+1)*16384)]
        sample_list.append(sample)
    if length_train % (16384//2) != 0:
        sample = data[-16384:]
        sample_list.append(sample)
    for j, audio in enumerate(sample_list):
        write_path = os.path.join(r'/no_backups/s1374/dataset16kHz_16384_overlap/clean_train', clean_sample[:-4] + '_' + str(j) + '.wav')
        wavfile.write(write_path, samplerate, audio)