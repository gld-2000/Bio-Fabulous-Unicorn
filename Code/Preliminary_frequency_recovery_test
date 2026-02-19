import mne
import numpy as np
import matplotlib.pyplot as plt

# keyboard to trigger keys: w = 1, d = 2, s = 3, a = 4
# loading the file
raw = mne.io.read_raw_bdf('/Users/helenakisic/Bio-Fabulous-Unicorn/Recordings/Flicker recordings/Wet_OSCAR+/UnicornRecorder_17_02_2026_16_58_20.bdf', preload = True)
print(raw.ch_names)  # check channel names 

# finding the trigger events
events = mne.find_events(raw, stim_channel='Status', mask=0xFF) 
#the mask is set to 0xOFF which in binary is 11111111, which means it will only keep the last 8 bits of the status channel value and throw away the rest
# we want to do this because the status channel is a 24-bit number that carries technical information about the recording system that we are not interested in
print("Events found:", events)
print("Unique trigger codes:", np.unique(events[:, 2]))

# define trigger codes 
event_id = {
    'upper':  1,   # 15 Hz
    'right':  2,   # 6 Hz
    'lower':  3,   # 10 Hz
    'left':   4,   # 7.5 Hz
}

# creating the epochs
# tmin = -0.5 starts half a second before trigger because of reaction time delay
# tmax = 9.5  ends 0.5s before the 10s stimulus ends
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=-0.5,
    tmax=9.5,
    baseline=None,   
    preload=True
)
print(epochs)

# crop epochs to focus on the stimulus period, excluding the initial reaction time and potential SSVEP stabilization phase
# skip first 1s after trigger 
# analyze from +1.0s to +9.5s = about 8.5s of clean signal
epochs_clean = epochs.copy().crop(tmin=1.0, tmax=9.5)

# power spectral density analysis to identify peaks at target frequencies (6, 7.5, 10, 15 Hz) in the occipital channels (channels 6, 7, 8)
conditions = {
    'upper (15 Hz)': 'upper',
    'right (6 Hz)':  'right',
    'lower (10 Hz)': 'lower',
    'left (7.5 Hz)': 'left',
}

# occipital channels are most relevant for SSVEP -> in Unicorn Hybrid Black these are channels 6, 7 and 8
occipital_channels = ['EEG 6', 'EEG 7', 'EEG 8']  

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, (label, cond) in enumerate(conditions.items()):
    # compute PSD for this condition
    psd = epochs_clean[cond].compute_psd(
        method='welch',
        fmin=1,
        fmax=30,
        picks=occipital_channels
    )
    
    # average across epochs and channels
    psd_data, freqs = psd.get_data(return_freqs=True)
    psd_mean = psd_data.mean(axis=(0, 1))  
    
    # plot
    axes[i].plot(freqs, psd_mean)
    axes[i].set_title(label)
    axes[i].set_xlabel('Frequency (Hz)')
    axes[i].set_ylabel('Power (µV²/Hz)')
    axes[i].axvline(x=float(label.split('(')[1].split(' ')[0]),
                    color='red', linestyle='--', label='target freq')
    axes[i].legend()
    axes[i].set_xlim(1, 30)

plt.tight_layout()
plt.suptitle('SSVEP Power Spectral Analysis', y=1.02, fontsize=14)
plt.show()
