import torchaudio
from utils import *
from natsort import natsorted

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dir, noisy_dir, threshold, transformation='real', name=False, ds_type='demand'):
        self.name = name
        self.threshold = threshold
        self.transformation = transformation
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.clean_wav_name = os.listdir(clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)
        self.noisy_wav_name = os.listdir(noisy_dir)
        self.noisy_wav_name = natsorted(self.noisy_wav_name)
        self.ds_type = ds_type
        assert ds_type in ['demand', 'timit']
        if ds_type == 'timit':
            if 'train' in clean_dir:
                self.module = 11579
            if 'test' in clean_dir:
                self.module = 4221

    def __len__(self):
        return len(self.noisy_wav_name)

    def __getitem__(self, idx):
        if self.ds_type == 'timit':
            noisy_wav_name = self.noisy_wav_name[idx]
            name_list = noisy_wav_name.split('_')
            clean_wav_name = '_'.join([name_list[0], name_list[1], name_list[2], name_list[-1]])
            clean_wav_path = os.path.join(self.clean_dir, clean_wav_name)
        else:
            clean_wav_path = os.path.join(self.clean_dir, self.clean_wav_name[idx])
            wav_name = self.clean_wav_name[idx]
        noisy_wav_path = os.path.join(self.noisy_dir, self.noisy_wav_name[idx])
        clean_ds, _ = torchaudio.load(clean_wav_path)
        noisy_ds, _ = torchaudio.load(noisy_wav_path)
        wav_length = clean_ds.size(1)

        if self.transformation == 'real':
            batch = transform(clean_ds, noisy_ds, self.threshold)
            if self.name:
                return batch, wav_name
            else:
                return batch
        else:
            return clean_ds, noisy_ds


def transform(target_audio, noisy_audio, threshold, n_fft=511, hop_len=64, size=(256, 256)):
    target_audio = torch.squeeze(target_audio)
    noisy_audio = torch.squeeze(noisy_audio)

    target_audio_stft_to_stack = stft(target_audio, n_fft, hop_len, n_fft, 'hamm', True, True,
                                      True)
    noisy_audio_stft_to_stack = stft(noisy_audio, n_fft, hop_len, n_fft, 'hamm', True, True,
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
    target_audio_stft_mag = preprocess(target_audio_stft_mag).unsqueeze(0)
    noisy_audio_stft_mag = preprocess(noisy_audio_stft_mag).unsqueeze(0)

    return target_audio_stft, \
           noisy_audio_stft, \
           target_audio, \
           noisy_audio, \
           target_audio_stft_mag, \
           noisy_audio_stft_mag


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data(train_clean_dir, test_clean_dir, train_noisy_dir, test_noisy_dir, batch_size, threshold, n_cpu, ds_type):
    torchaudio.set_audio_backend("sox_io")         # in linux

    Demand_test = DemandDataset(test_clean_dir, test_noisy_dir, threshold, 'real', ds_type=ds_type)
    Demand_train = DemandDataset(train_clean_dir, train_noisy_dir, threshold, 'real', ds_type=ds_type)

    train_dataset = torch.utils.data.DataLoader(dataset=Demand_train, batch_size=batch_size, shuffle=True,
                                                drop_last=True, worker_init_fn=seed_worker, num_workers=n_cpu,
                                                pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(dataset=Demand_test, batch_size=batch_size, shuffle=False,
                                               drop_last=True, worker_init_fn=seed_worker, num_workers=n_cpu,
                                               pin_memory=True)

    return train_dataset, test_dataset

