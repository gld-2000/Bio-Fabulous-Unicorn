import mne
import numpy as np
from sklearn.cross_decomposition import CCA
from collections import deque
import time

#parameters
fs = 250                #Sampling rate
window_length = 2.0     #seconds
step_size = 0.5         #seconds
channels = ['Oz','PO8','PO7','Pz'] 
freqs = [6, 7.5, 10, 15]  #SSVEP frequencies
harmonics = 2  #number of harmonics to include
smoothing_num = 4  #number of windows to ensure smoothness in the cursor movement 
#Keep the last number (example: 4) of predictions and then send them together to control the cursor movement.
#This is to ensure that the cursor movement is smooth and not jittery due to noise in the EEG signal.
#The majority vote of the last 5 predictions will determine the final direction of the cursor movement.

#Map frequency to direction
freq2dir = {15: "UP", 6: "RIGHT", 10: "DOWN", 7.5: "LEFT"}

def generate_reference(freq, fs, window_len, harmonics=2):
    t = np.arange(0, window_len, 1/fs) #time_vector from 0 to window_length with step size = 1 / sampling_rate
    ref = []
    for h in range(1, harmonics+1):
        ref.append(np.sin(2*np.pi*freq*h*t))
        ref.append(np.cos(2*np.pi*freq*h*t))
    return np.array(ref).T  #shape: (samples, 2*harmonics)
    #transpose 


def detect_ssvep(eeg_window, fs):
    """Detect frequency using CCA"""
    scores = []
    for f in freqs:
        window_duration = eeg_window.shape[1] / fs
        ref = generate_reference(f, fs, window_duration, harmonics)
        cca = CCA(n_components=1)
        cca.fit(eeg_window.T, ref)
        U, V = cca.transform(eeg_window.T, ref)
        corr = np.corrcoef(U.T, V.T)[0,1]
        scores.append(corr)
    return freqs[np.argmax(scores)]
#sklearn.cross_decomposition.CCA expects input shape like this:(n_samples, n_features(channels here))

#(Offline test)
raw = mne.io.read_raw_bdf("F:\\lea\\back to study\\2nd semester\\Biofablab\\Git codes\\Recordings\\Flicker recordings\\Wet_OSCAR-\\UnicornRecorder_17_02_2026_17_04_15.bdf", preload=True)

raw.pick('eeg')

#Rename hardware channels to anatomical names
mapping = {
    'EEG 1': 'Fz',
    'EEG 2': 'C3',
    'EEG 3': 'Cz',
    'EEG 4': 'C4',
    'EEG 5': 'Pz',
    'EEG 6': 'PO7',
    'EEG 7': 'Oz',
    'EEG 8': 'PO8'
}

raw.rename_channels(mapping)

#pick only occipital/parietal channels for SSVEP
raw.pick(['Pz','PO7','Oz','PO8'])

#raw.filter(5, 40)       
#raw.notch_filter(50)   


data = raw.get_data()     #shape: (n_channels, n_samples)
n_samples = data.shape[1]
window_samples = int(window_length * fs)
step_samples = int(step_size * fs)
frequency_list = []

#real-time sliding window simulation 
pred_buffer = deque(maxlen=smoothing_num)

print("Starting sliding window SSVEP detection simulation...")
count = 0
for start in range(0, n_samples - window_samples, step_samples):
    window = data[:, start:start+window_samples]
    freq_detected = detect_ssvep(window, fs)
    frequency_list.append(freq_detected)   #store detected frequencies for analysis
    pred_buffer.append(freq_detected)
    count += 1
    #smoothing
    direction = max(set(pred_buffer), key=pred_buffer.count)
    print(f"Detected freq: {freq_detected:.2f} Hz -> Direction: {freq2dir[direction]}")
    
    #Simulate real-time update every step_size
    time.sleep(step_size)

print(f"Total windows processed: {count}")
print(frequency_list)
#Total windows processed: 234
#29629 samples / 118.516 sec ≈ 250 Hz