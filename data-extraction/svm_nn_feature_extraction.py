from helper_functions import splitSignal, pad_or_trim_mfcc

import numpy as np
import librosa as lb
from glob import glob
import os
import joblib

def main():
    audio_dir = "./data/raw/"  # directory containing audio files
    meta_path = "./data/raw/audioMNIST_meta.txt"  # path to the metadata 
    num_mfcc = 25  # number of MFCC to extract
    n_segments = 1  # how many segements to split audio files
    sample_rate = 16000  # sampling rate

    file_paths = glob(audio_dir + "**/*.wav", recursive=True)
    file_paths.sort()

    print(f"Found {len(file_paths)} audio files.")
    print("Starting averaged MFCC extraction...")

    features = []

    for idx, fpath in enumerate(file_paths):
        audio_data, _ = lb.load(fpath, sr=sample_rate)  # samples the .wav file and puts it into time series

        segments = splitSignal(audio_data, n_segments)  # segmentation of the audio data

        target = int(os.path.basename(fpath).split('_')[0])  # grab target (digit) label from the filename

        for seg in segments:
            mfcc = lb.feature.mfcc(y=seg, sr=sample_rate, n_mfcc=num_mfcc)
            mfcc_avg = np.mean(lb.amplitude_to_db(np.abs(mfcc), ref=np.max), axis=1)

            features.append((mfcc_avg, target))

        if (idx + 1) % 500 == 0 or (idx + 1) == len(file_paths):
            print(f"Processed {idx + 1}/{len(file_paths)} files.")

    print(f"Extracted and averaged MFCCs for {len(features)} samples. Converting to arrays...")

    X = np.stack([feat for feat, _ in features])  # Shape (n_samples, n_mfcc)
    y = np.array([label for _, label in features])

    print(f"Final data shape: X = {X.shape}, y = {y.shape}. Saving to disk...")

    joblib.dump((X, y), './data/processed/avg_mfcc_data.pkl')

    print("Averaged MFCC data saved successfully.")

if __name__ == "__main__":
    main()