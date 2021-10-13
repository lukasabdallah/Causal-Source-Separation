from scipy.io.wavfile import read, write
from scipy import signal
import os
import pandas as pd
import numpy as np
from natsort import natsorted


def create_csv(path, cut_duration, overlap, keep_last):
    csv_data = pd.DataFrame(
        columns=['file_name', 'noise_type', 'start', 'end', "ori_length"])
    clean_dir = os.path.join(path, 'clean')
    noisy_dir = os.path.join(path, 'demand')
    reverberant_dir = os.path.join(path, 'reverberant')
    # noise_types = ['demand', 'reverberant']
    # noise_dirs = []
    # for noise_type in noise_types:
    #     noise_dirs.append(os.path.join(path, noise_type))

    clean_file_set = os.listdir(clean_dir)
    clean_file_set = natsorted(clean_file_set)
    # lost_samples_count = 0
    for file_name in clean_file_set:
        clean_path = os.path.join(clean_dir, file_name)
        noisy_path = os.path.join(noisy_dir, file_name)
        reverberant_path = os.path.join(reverberant_dir, file_name)
        sr_reverberant, reverberant_data = read(reverberant_path)
        sr_noisy, noisy_data = read(noisy_path)
        sr_clean, clean_data = read(clean_path)

        assert sr_reverberant == sr_noisy == sr_clean == 16000
        assert len(noisy_data) == len(clean_data) == len(reverberant_data)
        len_data = len(clean_data)
        # if len_data < cut_duration:
        #     lost_samples_count += 1
        #     print(lost_samples_count)       # ideally padding
        #     continue

        split_num = len_data // int(cut_duration * (1 - overlap))
        for i in range(split_num - 1):
            start = int(i * (cut_duration * (1 - overlap)))
            end = int(start + cut_duration)
            csv_data = csv_data.append([{'file_name': file_name, 'noise_type': "demand", 'start': start, 'end': end,
                                         'ori_length': len_data}], ignore_index=True)
            csv_data = csv_data.append([{'file_name': file_name, 'noise_type': 'reverberant', 'start': start, 'end': end,
                                         'ori_length': len_data}], ignore_index=True)

        if keep_last and len_data % int(cut_duration * (1 - overlap)) != 0:
            start = len_data - cut_duration
            end = len_data
            csv_data = csv_data.append([{'file_name': file_name, 'noise_type': "demand", 'start': start, 'end': end,
                                         'ori_length': len_data}], ignore_index=True)

            csv_data = csv_data.append([{'file_name': file_name, 'noise_type': "reverberant", 'start': start, 'end': end,
                                         'ori_length': len_data}], ignore_index=True)

    return csv_data


def main():
    cut_duration = 16128
    overlap = 0.5
    for arg in ["test", "train"]:
        path = f'Demand_16kHz/{arg}'
        target_path = f'Demand_16kHz/{arg}/cut{cut_duration}_ov{overlap}.csv'
        csv_data = create_csv(path, cut_duration=cut_duration,
                              overlap=overlap, keep_last=True)
        csv_data.to_csv(target_path)


if __name__ == '__main__':
    main()
