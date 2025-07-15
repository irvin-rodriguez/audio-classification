from helper_functions import splitSignal, pad_or_trim_mfcc

import numpy as np
import librosa as lb
from glob import glob
import os
import joblib

def main():
    audio_dir = "./data/raw/"  # directory containing audio files
    # meta_path = "./data/raw/audioMNIST_meta.txt"  # path to the metadata 
    num_mfcc = 25  # number of MFCC to extract
    n_segments = 1  # how many segements to split audio files
    sample_rate = 16000  # sampling rate
    target_time_steps = 32  # largest is 32 so we are not triming any audio files, just padding

    file_paths = glob(audio_dir + "**/*.wav", recursive=True)
    file_paths.sort()

    print(f"{len(file_paths)} audio files found.")
    print("Starting MFCC extraction...")

    cnn_features = []  # list of (mfcc_matrix, target) tuples
    for idx, fpath in enumerate(file_paths):
        audio_data, _ = lb.load(fpath, sr=sample_rate)

        segments = splitSignal(audio_data, n_segments)  # 1 segmenet = full audio

        target = int(os.path.basename(fpath).split('_')[0])

        for seg in segments:
            mfcc = lb.feature.mfcc(y=seg, sr=sample_rate, n_mfcc=num_mfcc)
            mfcc_db = lb.amplitude_to_db(np.abs(mfcc), ref=np.max)
            
            # Instead of flattening or averaging, store the full matrix with label
            cnn_features.append((mfcc_db, target))

        if (idx + 1) % 500 == 0 or (idx + 1) == len(file_paths):
            print(f"Processed {idx + 1}/{len(file_paths)} files.")
    
    print(f"Extracted MFCCs for {len(cnn_features)} segments. Starting padding...")

    padded_mfcc_list = []
    label_list = []

    for idx, (mfcc, label) in enumerate(cnn_features):
        mfcc_fixed = pad_or_trim_mfcc(mfcc, target_time_steps)
        padded_mfcc_list.append(mfcc_fixed)
        label_list.append(label)

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(cnn_features):
            print(f"Padded {idx + 1}/{len(cnn_features)} samples.")

    # Convert to numpy arrays
    X = np.stack(padded_mfcc_list)  # Shape (n_samples, n_mfcc, time_steps)
    y = np.array(label_list)

    print(f"Final data shape: X = {X.shape}, y = {y.shape}. Saving to disk...")

    # save the target and labels
    joblib.dump((X, y), './data/processed/cnn_mfcc_data.pkl')

    print("Data saved successfully.")

if __name__ == "__main__":
    main()