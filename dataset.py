import torch
import math
import torchaudio
import os
from utils import *
import pandas as pd
from natsort import natsorted
# import matplotlib.pyplot as plt
# import seaborn as sns
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_file, threshold, transformation='real', name=False):
        self.name = name
        self.threshold = threshold
        self.transformation = transformation
        self.data_dir = data_dir
        self.csv_file = pd.read_csv(f"{self.data_dir}/{csv_file}")

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        file_name = self.csv_file['file_name'][idx]
        start = self.csv_file['start'][idx]
        end = self.csv_file['end'][idx]
        length = self.csv_file['ori_length'][idx]
        noise_type = self.csv_file['noise_type'][idx]
        clean_wav_path = f'{self.data_dir}/clean/{file_name}'
        noisy_wav_path = f'{self.data_dir}/{noise_type}/{file_name}'
        clean_ds, _ = torchaudio.load(clean_wav_path)
        noisy_ds, _ = torchaudio.load(noisy_wav_path)
        clean_ds = clean_ds[:, start: end]
        noisy_ds = noisy_ds[:, start: end]

        if self.transformation == 'real':
            batch = transform(clean_ds, noisy_ds, noise_type, self.threshold)
            if self.name:
                return batch, length, file_name
            else:
                return batch
        else:
            return clean_ds, noisy_ds


def transform(target_audio, noisy_audio, noise_type, threshold, n_fft=511, hop_len=63, size=(256, 256)):
    target_audio = torch.squeeze(target_audio)
    noisy_audio = torch.squeeze(noisy_audio)

    target_audio = normalize_tensor(target_audio)
    noisy_audio = normalize_tensor(noisy_audio)

    target_audio_stft_to_stack = stft(target_audio, n_fft, hop_len, n_fft, 'hamm', True, True,
                                      True)
    noisy_audio_stft_to_stack = stft(noisy_audio, n_fft, hop_len, n_fft, 'hamm', True, True,
                                     True)
    target_audio_stft_mag = torch.abs(target_audio_stft_to_stack)
    # target_audio_stft_mag = torch.sqrt(target_audio_stft_to_stack[0]**2 + target_audio_stft_to_stack[1]**2)
    target_audio_stft_phase = torch.angle(target_audio_stft_to_stack)
    noisy_audio_stft_mag = torch.abs(noisy_audio_stft_to_stack)
    noisy_audio_stft_phase = torch.angle(noisy_audio_stft_to_stack)

    target_audio_stft_mag = 20 * torch.log10(target_audio_stft_mag)
    noisy_audio_stft_mag = 20 * torch.log10(noisy_audio_stft_mag)

    target_audio_stft_mag = torch.sub(
        target_audio_stft_mag, torch.max(target_audio_stft_mag))
    target_audio_stft_mag = scale_mag(target_audio_stft_mag, size, threshold)
    target_audio_stft_mag = normalise_spect(target_audio_stft_mag)

    noisy_audio_stft_mag = torch.sub(
        noisy_audio_stft_mag, torch.max(noisy_audio_stft_mag))
    noisy_audio_stft_mag = scale_mag(noisy_audio_stft_mag, size, threshold)
    noisy_audio_stft_mag = normalise_spect(noisy_audio_stft_mag)

    # stft with normalized mag
    noisy_audio_stft = conv_complex_after_stft(
        noisy_audio_stft_mag, noisy_audio_stft_phase, False)
    target_audio_stft = conv_complex_after_stft(
        target_audio_stft_mag, target_audio_stft_phase, False)
    # '''Convert the magnitude to scale (-1, 1) for the generator and discriminator'''
    target_audio_stft_mag = preprocess(target_audio_stft_mag).unsqueeze(0)
    noisy_audio_stft_mag = preprocess(noisy_audio_stft_mag).unsqueeze(0)

    return target_audio_stft, \
        noisy_audio_stft, \
        target_audio, \
        noisy_audio, \
        target_audio_stft_mag, \
        noisy_audio_stft_mag, \
        noise_type


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data(data_dir, csv_file, batch_size, threshold, n_cpu):
    # torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
    # torchaudio.set_audio_backend("soundfile")       # in windows
    torchaudio.set_audio_backend("sox_io")         # in linux
    train_dir = f"{data_dir}/train"
    test_dir = f"{data_dir}/test"
    Demand_test = DemandDataset(
        train_dir, csv_file, threshold, 'real')
    Demand_train = DemandDataset(
        test_dir,  csv_file, threshold, 'real')

    train_dataset = torch.utils.data.DataLoader(dataset=Demand_train, batch_size=batch_size,
                                                shuffle=True, drop_last=True, worker_init_fn=seed_worker, num_workers=n_cpu, pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(dataset=Demand_test, batch_size=batch_size, shuffle=False,
                                               drop_last=True, worker_init_fn=seed_worker, num_workers=n_cpu, pin_memory=True)

    return train_dataset, test_dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy
    data_dir = 'DEMAND_16KHz'
    csv_file = 'cut16128_ov0.5.csv'
    train_set, test_set = load_data(
        data_dir, csv_file, 4, -60, 1)
    for idx, sample in enumerate(test_set):
        # complex_noisy = sample[0]
        # clean_audio = sample[2][0]
        # complex_clean = sample[5]
        # mag = torch.abs(complex_noisy)

        f = np.ogrid[0:8000:256j]
        t = np.ogrid[0:1:256j]
        fig = plt.figure(figsize=[6.4, 6.8])
        denoised = fig.add_subplot(211)
        plt.pcolormesh(t, f, sample[4][1][0], shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        noisy = fig.add_subplot(212, sharex=denoised)
        plt.pcolormesh(t, f, sample[5][1][0], shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        denoised.title.set_text('clean')
        noisy.title.set_text(sample[6][1])
        plt.show()
        # for noisy, clean in zip(complex_noisy, complex_clean):
        #     #phase = torch.sin(clean)
        #     noisy_spc = sns.heatmap(noisy)
        #     plt.show()
        if idx == 4:
            break
