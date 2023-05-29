import base64
from fastapi import FastAPI
import math
from cnn import CNN
from enum import Enum
import numpy as np
import torch
import torchaudio
import librosa
import io

# Define params
SAMPLE_RATE = 22050
DURATION = 3.5 # desired duration (used in training)
NUM_SAMPLES = 22050 * DURATION
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

# Load the model
MODEL = CNN()
MODEL =  torch.load(r'models\model.pt', map_location=torch.device("cpu"))
MODEL.eval()
# set device
DEVICE = torch.device('cpu')
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
 
def onset_split_prediction(signal, sr, metronome_type):
    if metronome_type == 'auto':
        # Detect onsets using the spectral flux and adaptive threshold
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

        # Detect onsets using the spectral flux and adaptive threshold
        onsets = librosa.onset.onset_detect(onset_envelope=spectral_flux,
                                                hop_length=SAMPLE_RATE,
                                                sr=sr,
                                                units='frames',
                                                backtrack=False,
                                                pre_max = 10, post_max = 10, pre_avg = 50, post_avg = 50, delta = 0.1,
                                                wait=0)

        # convert to samples
        onsets = librosa.frames_to_samples(onsets)

        # You might want to adjust these parameters
        queue_length = 5  # Number of recent onset distances to consider
        initial_threshold = SAMPLE_RATE  # Initial threshold until you have enough onsets to compute the moving average


        # filter by moving average
        recent_distances = []

        filtered_onsets = [onsets[0]]  # start with the first onset

        for i in range(1, len(onsets)):
            if len(recent_distances) < queue_length:
                threshold = initial_threshold
            else:
                threshold = np.mean(recent_distances)

            if onsets[i] - filtered_onsets[-1] >= threshold:
                filtered_onsets.append(onsets[i])
                if len(filtered_onsets) > 1:  # dont add a distance until there are at least two onsets
                    recent_distances.append(filtered_onsets[-1] - filtered_onsets[-2])

            # If the list of recent distances is too long, remove the oldest distance
            if len(recent_distances) > queue_length:
                recent_distances.pop(0) 

        # Shift all onsets by a tenth of the average distance
        filtered_onsets = filtered_onsets - np.mean(np.diff(filtered_onsets))/10

        # if first onset went under 0, add it back to 0
        if filtered_onsets[0] < 0: filtered_onsets[0] =0

        filtered_onsets = np.array(filtered_onsets)

        # convert back to frames
        filtered_onsets = librosa.samples_to_frames(filtered_onsets)
    else:
        m_signal, sr = librosa.load(path = r'metronomes\metronome' + metronome_type + ".wav", sr = SAMPLE_RATE)
        m_signal = m_signal[:len(signal)] # cut metronome audio to length of analysis file
        onsets = librosa.onset.onset_detect(y = m_signal, sr = SAMPLE_RATE, units='samples')

        spike_threshold = 0.2  # adjust as needed

        # get the first spike sammple
        first_spike_sample = np.argmax(np.abs(signal) > spike_threshold)

        # Calculate the difference between the first onset and the first spike in samples
        sample_diff = first_spike_sample - onsets[0]

        # Shift all onsets by the sample difference - 
        onsets = onsets + sample_diff -  math.trunc(SAMPLE_RATE * 0.25)

        # Filter out onsets that fall outside the valid range of the signal
        onsets = onsets[(onsets >= 0) & (onsets < len(signal))]

       # Get the maximum amplitude of each slice of audio
        max_amplitudes = [np.max(np.abs(signal[onsets[i]:onsets[i + 1]])) for i in range(len(onsets) - 1)]

        # Create an array to store the filtered onsets
        filtered_onsets = []

        # Check if the first slice has significant audio (adjust the threshold as needed)
        if max_amplitudes[0] > 0.15:  #silence threshold
            filtered_onsets.append(onsets[0])

        # For each slice of audio, starting from the second one
        for i in range(1, len(max_amplitudes)):
            # If the amplitude of the previous slice is not higher than the current one
            if max_amplitudes[i - 1] <= max_amplitudes[i]:
                # Keep the onset that started the current slice
                filtered_onsets.append(onsets[i])

        # Convert the list back to a numpy array
        filtered_onsets = np.array(filtered_onsets)

        # Convert back to frames
        filtered_onsets = librosa.samples_to_frames(filtered_onsets)

        # filtered_onsets = librosa.samples_to_frames(onsets)


    # Plot the waveform with onsets
    # plot_onsets(signal, sr, filtered_onsets)

    # Convert onsets to samples
    filtered_onsets = librosa.frames_to_samples(filtered_onsets)

    # Split the signal by onsets
    signal_segments = split_signal_by_onsets(signal=signal, filtered_onsets=filtered_onsets)
    signal_segments = [preprocess_split(x) for x in signal_segments]

    print('SEGMENTS: ' + str(len(signal_segments)))

    predictions = [predict(x.cpu()) for x in signal_segments]

    time_onsets = librosa.samples_to_time(filtered_onsets, sr=sr)

    chord_onsets = dict(zip(time_onsets.tolist(), predictions))

    return chord_onsets

app = FastAPI()

@app.post("/analyse/")
async def analyse(audio_data_json: dict):
    try:
        base64_audio_data = audio_data_json['AudioData']
        metronome_type = audio_data_json['MetronomeType']
        print('METRONOME: ' + metronome_type)
        audio_data = base64.b64decode(base64_audio_data)
        results = analyse_audio_file(io.BytesIO(audio_data), metronome_type)  # Call Python function to analyse the audio file
        return results
    except Exception as e:
        print("Failed to process audio")
        print(e)
        return "Failed to process audio"

def analyse_audio_file(audio_bytes, metronome_type):
    signal, sr = librosa.load(audio_bytes, sr=SAMPLE_RATE)
    print('SIGNAL SIZE: ' + str(signal.size))
    predictions = onset_split_prediction(signal, sr, metronome_type)
    print(predictions)
    return predictions