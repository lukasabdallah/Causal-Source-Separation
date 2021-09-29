from scipy import signal
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import librosa
import torchaudio
import torch
import numpy as np
from utils import *

# torchaudio.set_audio_backend("sox_io")

source_path = "DEMAND_16KHz/test/reverberant/p232_002.wav"
noisy_path = "DEMAND_16KHz/test/reverberant/p232_002.wav"
noisy_d, _ = torchaudio.load(noisy_path)
data, sampling_rate = torchaudio.load(source_path)
# data = data[-16000:]
# data = data[-16000:]
# print(data[0][:16000].shape)
assert len(torch.squeeze(data)) == len(torch.squeeze(noisy_d))

data = torch.squeeze(data)[:16128]


window = torch.hamming_window(511)
window_x = np.arange(0, 511, 1)
# plt.plot(window_x, window)
# plt.show()


n_fft = 511
hop_length = 63


noisy_audio_stft_to_stack = torch.stft(data, n_fft=n_fft, hop_length=hop_length,
                                       win_length=n_fft, window=window, normalized=True, center=True, onesided=True, return_complex=True)
# print(noisy_audio_stft_to_stack.shape)
print(noisy_audio_stft_to_stack.unsqueeze(0).shape)
# Sxx = batch size ×N×T×2
# Sxx = batch size ×N×T if returncomplex = True


noisy_audio_stft_mag = torch.abs(noisy_audio_stft_to_stack)
# noisy_audio_stft_mag = librosa.power_to_db(noisy_audio_stft_mag)
noisy_audio_stft_mag = 20 * torch.log10(noisy_audio_stft_mag)

# scaling:
noisy_audio_stft_mag = torch.sub(
    noisy_audio_stft_mag, torch.max(noisy_audio_stft_mag))
size = (256, 256)
threshold = -90
noisy_audio_stft_mag = scale_mag(noisy_audio_stft_mag, size, threshold)

noisy_audio_stft_mag = normalise_spect(noisy_audio_stft_mag)

f = np.ogrid[0:8000:256j]
t = np.ogrid[0:1:256j]

print(noisy_audio_stft_mag)
plt.pcolormesh(t, f, noisy_audio_stft_mag, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

plt.show()
