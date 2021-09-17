from scipy.io.wavfile import read, write
from scipy import signal
import os
import pandas as pd
import numpy as np
from natsort import natsorted


def check_clean_name(clean_name):
    clean_name_split = clean_name.split('_')        # file names are usually split with "_"
    return len(clean_name_split)


def generate_clean_name(noisy_name, clean_name_len):
    noisy_name_split = noisy_name.split('_')
    clean_name = '_'.join([noisy_name_split[i] for i in range(clean_name_len)])
    # for i in range(clean_name_len):
    #     clean_name += noisy_name_split[i]
    if clean_name[-4:] != '.wav':
        clean_name += '.wav'
    return clean_name


def create_csv(path, cut_duration, overlap, keep_last):
    csv_data = pd.DataFrame(columns=['clean_name', 'noisy_name', 'start', 'end', "ori_length"])
    noisy_dir = os.path.join(path, 'noisy')
    clean_dir = os.path.join(path, 'clean')
    noisy_file_set = os.listdir(noisy_dir)
    noisy_file_set = natsorted(noisy_file_set)
    clean_file_set = os.listdir(clean_dir)
    clean_file_set = natsorted(clean_file_set)

    clean_file_example = clean_file_set[0]
    clean_name_len = check_clean_name(clean_file_example)
    for noisy_file in noisy_file_set:
        clean_file = generate_clean_name(noisy_file, clean_name_len)
        clean_path = os.path.join(clean_dir, clean_file)
        noisy_path = os.path.join(noisy_dir, noisy_file)
        sr, noisy_data = read(noisy_path)
        _, clean_data = read(clean_path)
        assert sr == 16000
        assert len(noisy_data) == len(clean_data)
        len_noisy_data = len(noisy_data)
        if len_noisy_data < cut_duration:       # ideally padding
            continue
        split_num = len_noisy_data // int(cut_duration * (1 - overlap))
        for i in range(split_num - 1):
            start = int(i * (cut_duration * (1 - overlap)))
            end = int(start + cut_duration)
            csv_data = csv_data.append([{'clean_name': clean_file, 'noisy_name': noisy_file, 'start': start, 'end': end,
                                         'ori_length': len_noisy_data}], ignore_index=True)
        if keep_last and len_noisy_data % int(cut_duration * (1 - overlap)) != 0:
            start = len_noisy_data - cut_duration
            end = len_noisy_data
            csv_data = csv_data.append([{'clean_name': clean_file, 'noisy_name': noisy_file, 'start': start, 'end': end,
                                         'ori_length': len_noisy_data}], ignore_index=True)

    return csv_data


def main():
    path = r'/no_backups/s1374/Demand_16kHz/train'
    target_path = r'/no_backups/s1374/Demand_16kHz/train/csvfile_keep_last.csv'
    csv_data = create_csv(path, cut_duration=16384, overlap=0.5, keep_last=True)
    csv_data.to_csv(target_path)

    # path = r'/no_backups/s1374/Demand_16kHz/test'
    # target_path = r'/no_backups/s1374/Demand_16kHz/test/csvfile.csv'
    # csv_data = create_csv(path, cut_duration=16384, overlap=0, keep_last=False)
    # csv_data.to_csv(target_path)


if __name__ == '__main__':
    main()
