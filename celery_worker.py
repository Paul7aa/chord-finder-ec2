import io
import math
import os
import urllib.parse
from celery import Celery
from cnn import CNN
from enum import Enum
import numpy as np
import torch
import torchaudio
import librosa
import base64
import sys

# Define params
SAMPLE_RATE = 22050
DURATION = 3.5 # desired duration (used in training)
NUM_SAMPLES = 22050 * DURATION
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

# Load the model
MODEL =  CNN()
MODEL.load_state_dict(torch.load('models/model.pt', map_location=("cpu")))
MODEL.eval()
# set device
DEVICE = torch.device('cpu')

# Define transformation
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
sample_rate=SAMPLE_RATE,
n_fft=N_FFT,
hop_length=HOP_LENGTH,
n_mels=N_MELS,
normalized=True
)

mel_spectrogram = mel_spectrogram.to(DEVICE)

class CLASS_MAPPING(Enum):
    Am = 0
    AsharpM = 1
    Bb = 2
    Bm = 3
    Bdim = 4
    Cdim = 5
    C = 6
    Csharp = 7
    Dm = 8
    DsharpM = 9
    Em = 10
    Fm = 11
    F = 12
    Fsharp = 13
    G = 14
    Gsharp = 15

# PREPROCESSING FUNCTIONS (FOR SPLIT)------------------------------------------

# preprocessing functions

# function to change sample rate to target rate
def _resample(signal, sr):
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        resampler.to(DEVICE)
        signal = resampler(signal)
    return signal

# function to transform audio to monochannel
def _mix_down(signal):
    if signal.shape[0] > 1: # signal.shape[0] = number of channels
        signal = torch.mean(signal,
                            dim=0,
                            keepdim = True)
    return signal

# function to cut audio samples to desired duration
def _cut(signal):
    if signal.shape[1] > NUM_SAMPLES:
        signal = signal[:, :math.floor(NUM_SAMPLES)]
    return signal

def _right_pad(signal):
    signal_length = signal.shape[1]
    if signal_length < NUM_SAMPLES:
        num_missing_samples = NUM_SAMPLES - signal_length
        last_dim_padding = (0, math.floor(num_missing_samples))
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

def _normalize_data(signal):
    #mean-and-variance normalization
    mean = torch.mean(signal)
    std = torch.std(signal)
    return (signal - mean) / std

#PREPROCESSING FUNCTIONS (FOR SEQUENCE)------------------------------------------
def split_signal_by_onsets(signal, filtered_onsets):
    # Initialize an empty list to store the signal segments
    signal_segments = []

    # Iterate through the filtered onsets
    for i in range(len(filtered_onsets) - 1):
        # Compute the start and end sample indices for each segment
        start_sample = filtered_onsets[i] - 22050
        end_sample = filtered_onsets[i + 1] - 22050

        # Extract the segment from the original signal
        segment = signal[start_sample:end_sample]

        # Append the segment to the list
        signal_segments.append(segment)

    # Add the last segment from the last onset to the end of the signal
    start_sample = filtered_onsets[-1] - 22050
    end_sample = len(signal)
    segment = signal[start_sample:end_sample]
    signal_segments.append(segment)

    return signal_segments
def preprocess_split(signal):
    # convert to torch tensor
    signal = torch.tensor(signal) 
    signal = torch.unsqueeze(signal, 0)  # Add an extra dimension
    # preprocess
    signal = signal.to(DEVICE)
    signal = _resample(signal, SAMPLE_RATE)
    signal = _mix_down(signal)
    signal = _cut(signal)
    signal = _right_pad(signal)
    # Apply the Mel Spectrogram transformation
    signal = mel_spectrogram(signal)
    signal = _normalize_data(signal)
    return signal

def predict(preprocessed_data):
    # Make the prediction
    with torch.no_grad():
        prediction = MODEL(preprocessed_data)
        predicted_class_index = torch.argmax(prediction).item()
        predicted_chord_label = CLASS_MAPPING(predicted_class_index).name

    return predicted_chord_label
 
def onset_split_prediction(signal, sr):
    # Apply exponential moving average filter to smooth the signal
    alpha = 0.0005 # Smoothing factor (adjust as needed)
    smoothed_signal = np.zeros_like(signal)
    smoothed_signal[0] = signal[0]
    for i in range(1, len(signal)):
        smoothed_signal[i] = alpha * signal[i] + (1 - alpha) * smoothed_signal[i - 1]

    # Compute the magnitude spectrogram
    magnitude = np.abs(librosa.stft(smoothed_signal))

    # Compute the spectral flux as the squared difference between consecutive frames
    spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)

    # Calculate the mean and standard deviation of the spectral flux
    mean_flux = np.mean(spectral_flux)
    std_flux = np.std(spectral_flux)

    # Set the threshold as a multiple of the standard deviation above the mean
    threshold_multiplier = 2.5 # Adjust the multiplier as needed
    threshold = mean_flux + (std_flux * threshold_multiplier)

    # Detect onsets using the spectral flux and adaptive threshold
    onsets = librosa.onset.onset_detect(onset_envelope=spectral_flux,
                                        hop_length=SAMPLE_RATE,
                                        sr=sr,
                                        units='frames',
                                        backtrack=False,
                                        pre_max = 10, post_max = 10, pre_avg = 50, post_avg = 50, delta = 0.1,
                                        wait=0)

    # Get the beat locations
    _, beat_frames = librosa.beat.beat_track(y=signal, sr=sr, hop_length=HOP_LENGTH)
    
    # Calculate the average duration between beats
    average_beat_duration = np.mean(np.diff(beat_frames))

    # Calculate the minimum distance based on tempo
    min_distance = int(average_beat_duration / 0.5)  # Adjust the division factor as needed

    # Filter the onsets based on their values and minimum distance
    filtered_onsets = []
    last_onset = None
    for onset in onsets:
        if last_onset is None or (onset - last_onset) >= min_distance:
            if spectral_flux[onset] > threshold:
                filtered_onsets.append(onset)
                last_onset = onset

    # Plot the waveform with onsets
    # plot_onsets(signal, sr, filtered_onsets)

    # Convert onsets to samples
    filtered_onsets = librosa.frames_to_samples(filtered_onsets)

    # Split the signal by onsets
    signal_segments = split_signal_by_onsets(signal=signal, filtered_onsets=filtered_onsets)

    signal_segments = [preprocess_split(x) for x in signal_segments]

    print(len(signal_segments))

    predictions = [predict(x.cpu()) for x in signal_segments]

    time_onsets = librosa.samples_to_time(filtered_onsets, sr=sr)

    chord_onsets = dict(zip(time_onsets.tolist(), predictions))

    return chord_onsets

BROKER_URL = os.environ.get('REDISCLOUD_URL')
celery_app = Celery("audio_analysis_celery", broker=BROKER_URL)

@celery_app.task
def process_audio(audio_data_json):
    base64_audio_data = audio_data_json['AudioData']
    audio_data = base64.b64decode(base64_audio_data)
    signal, sr = librosa.load(io.BytesIO(audio_data), sr=SAMPLE_RATE)
    predictions = onset_split_prediction(signal, sr)
    return predictions
