import librosa
import soundfile as sf
import os


def main():
    source_path = "./28spk/train/reverberant"
    target_path = "./DEMAND_16KHz/train/reverberant/"
    dir = os.path.dirname(target_path)
    if not os.path.exists(target_path):
        os.makedirs(dir)
    files = os.listdir(source_path)
    files_len = len(files)
    counter = 0
    for file in files:
        counter += 1
        data, sampling_rate = librosa.load(
            f"{source_path}/{file}", sr=16000)
        sf.write(f"{target_path}/{file}", data, sampling_rate)
        if counter % 50 == 0:
            print(f"{counter}/{files_len}")


if __name__ == '__main__':
    main()
