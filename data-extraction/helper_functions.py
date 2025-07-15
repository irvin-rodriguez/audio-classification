import numpy as np

def splitSignal(signal, n_segments):
    """
    Splits an 1D audio signal into a defined number of equal segments.
    Data augmentation technique that treats each individual segment as a seperate sample.
    """

    signal_length = len(signal)
    segment_length = int(np.ceil(signal_length / n_segments))  # size that each segment needs to be
    segment_list = []

    for i in range(n_segments):
        start_index = i * segment_length  # starting index of current segment
        end_index = min((i + 1) * segment_length, signal_length)  # end index of current segment
        segment = signal[start_index:end_index]  # grab the segment
        segment_list.append(segment)  # add it to the list of segments

    return segment_list

def pad_or_trim_mfcc(mfcc, target_steps):
    if mfcc.shape[1] < target_steps:
        pad_width = target_steps - mfcc.shape[1]
        return np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return mfcc[:, :target_steps]
