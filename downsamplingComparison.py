from scipy.io.wavfile import read, write
import scipy.signal as sps
import librosa
import soundfile as sf
# from librosa import write_wav

# Your new sampling rate
new_rate = 16000

source_path = "28spk/test/reverberant/p232_002.wav"
# Read file
sampling_rate, data = read(source_path)
lib_data, lib_sampling_rate = librosa.load(
    source_path, sr=16000)
print(sampling_rate)
print(lib_sampling_rate)


# Resample data
number_of_samples = round(len(data) * float(new_rate) / sampling_rate)
data_resampled = sps.resample(data, number_of_samples)
data_decimated = sps.decimate(data, 3)

write("resampled_sps.wav", new_rate, data_resampled)
write("read_write_resampled.wav", new_rate, data)
write("decimated.wav", new_rate, data_decimated)

sf.write("lib.wav", lib_data, new_rate)

resampled_sps_sr, _ = read("resampled_sps.wav")
read_write_resampled_sr, _ = read("read_write_resampled.wav")
data_decimated_sr, _ = read("decimated.wav")

print(f"resampled_sps_sr = {resampled_sps_sr}")
print(f"read_write_resampled_sr = {read_write_resampled_sr}")
print(f"data_decimated_sr = {data_decimated_sr}")
