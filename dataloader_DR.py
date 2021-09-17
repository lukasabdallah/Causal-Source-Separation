import torch
import torchaudio
import os
from tools.utils_DR import *
from natsort import natsorted
import matplotlib.pyplot as plt
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dir, noisy_dir, csv_file, max_len, threshold, transformation='real'):
        self.max_len = max_len
        self.threshold = threshold
        self.transformation = transformation
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.clean_wav_name = os.listdir(clean_dir)
        self.csv_file = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        clean_file = self.csv_file['clean_name'][idx]
        noisy_file = self.csv_file['noisy_name'][idx]
        start = self.csv_file['start'][idx]
        end = self.csv_file['end'][idx]
        length = self.csv_file['ori_length'][idx]
        clean_wav_path = os.path.join(self.clean_dir, clean_file)
        noisy_wav_path = os.path.join(self.noisy_dir, noisy_file)
        clean_ds, _ = torchaudio.load(clean_wav_path)
        noisy_ds, _ = torchaudio.load(noisy_wav_path)
        clean_ds = clean_ds[:, start: end]
        noisy_ds = noisy_ds[:, start: end]
        wav_length = len(clean_ds[0])
        if self.transformation == 'real':
            batch = transform(clean_ds, noisy_ds, wav_length, clean_file, self.max_len, self.threshold)
            return batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]
        else:
            return clean_ds, noisy_ds, wav_length, clean_file


def transform(target_audio, noisy_audio, wav_length, wav_path, max_len, threshold, size=(512, 512)):
    target_audio = torch.squeeze(target_audio)
    noisy_audio = torch.squeeze(noisy_audio)
    overlap = overlap_calculation(wav_length)
    length_for_stft = (512 * (1023 - overlap) + overlap)
    target_audio = adapt_length_torch(target_audio[0: wav_length], wav_length, length_for_stft)
    noisy_audio = adapt_length_torch(noisy_audio[0: wav_length], wav_length, length_for_stft)
    target_audio_comp = torch.clone(target_audio)
    target_audio_comp = F.pad(target_audio_comp, (0, max_len - length_for_stft))
    noisy_audio = F.pad(noisy_audio, (0, max_len - length_for_stft))
    target_audio_to_stack = target_audio[0:length_for_stft]
    noisy_audio_to_stack = noisy_audio[0:length_for_stft]
    overlap_for_stft_to_stack = torch.tensor((1023 - np.int32(overlap)))
    target_audio_stft_to_stack = stft(target_audio_to_stack, 1023, overlap_for_stft_to_stack, 1023, 'hamm', False, True,
                                      True)
    noisy_audio_stft_to_stack = stft(noisy_audio_to_stack, 1023, overlap_for_stft_to_stack, 1023, 'hamm', False, True,
                                     True)

    target_audio_stft_mag = torch.abs(target_audio_stft_to_stack)
    target_audio_stft_phase = torch.angle(target_audio_stft_to_stack)
    noisy_audio_stft_mag = torch.abs(noisy_audio_stft_to_stack)
    noisy_audio_stft_phase = torch.angle(noisy_audio_stft_to_stack)

    target_audio_stft_mag = 20 * torch.log10(target_audio_stft_mag)
    target_audio_stft_mag = torch.sub(target_audio_stft_mag, torch.max(target_audio_stft_mag))
    target_audio_stft_mag = scale_mag(target_audio_stft_mag, size, threshold)
    target_audio_stft_mag = normalise_spect(target_audio_stft_mag)

    noisy_audio_stft_mag = 20 * torch.log10(noisy_audio_stft_mag)
    noisy_audio_stft_mag = torch.sub(noisy_audio_stft_mag, torch.max(noisy_audio_stft_mag))
    noisy_audio_stft_mag = scale_mag(noisy_audio_stft_mag, size, threshold)
    noisy_audio_stft_mag = normalise_spect(noisy_audio_stft_mag)

    noisy_audio_stft = conv_complex_after_stft(noisy_audio_stft_mag, noisy_audio_stft_phase, False)
    target_audio_stft = conv_complex_after_stft(target_audio_stft_mag, target_audio_stft_phase, False)
    # '''Convert the magnitude to scale (-1, 1) for the generator and discriminator'''
    target_audio_stft_mag = preprocess(target_audio_stft_mag).unsqueeze(0)
    noisy_audio_stft_mag = preprocess(noisy_audio_stft_mag).unsqueeze(0)

    return noisy_audio_stft_mag, \
           target_audio_stft_mag, \
           noisy_audio_stft, \
           target_audio_stft, \
           target_audio_comp, \
           noisy_audio, \
           overlap_for_stft_to_stack, \
           length_for_stft


def load_data(train_clean_dir, test_clean_dir, train_noisy_dir, test_noisy_dir, csv_train, csv_test, batch_size, max_len, threshold, n_cpu):
    # torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
    # torchaudio.set_audio_backend("soundfile")       # in windows
    torchaudio.set_audio_backend("sox_io")         # in linux
    Demand_train = DemandDataset(train_clean_dir, train_noisy_dir, csv_train, max_len, threshold, 'real')
    Demand_test = DemandDataset(test_clean_dir, test_noisy_dir, csv_test, max_len, threshold, 'real')
    train_dataset = torch.utils.data.DataLoader(dataset=Demand_train, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=n_cpu, pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(dataset=Demand_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=n_cpu, pin_memory=True)

    return train_dataset, test_dataset