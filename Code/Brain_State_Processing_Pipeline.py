#importing packages 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from pathlib import Path


samplingRate = 250 #Hz

#file paths
noGelRelaxed = pd.read_csv('./Recordings/No gel/Relaxed/UnicornRawDataRecorder_04_02_2026_15_07_130.csv')
noGelStimulated = pd.read_csv('./Recordings/No gel/Stimulated/UnicornRawDataRecorder_04_02_2026_15_22_320.csv')
gelRelaxed = pd.read_csv('./Recordings/With gel/Relaxed/UnicornRawDataRecorder_04_02_2026_16_04_170.csv')
gelStimulated = pd.read_csv('./Recordings/With gel/Stimulated/UnicornRawDataRecorder_04_02_2026_16_07_110.csv')

#drop unnecessary columns
dropCols = [" ACC X", " ACC Y", " ACC Z", " GYR X", " GYR Y", " GYR Z", " BAT"]
noGelRelaxed.drop(columns=dropCols, inplace=True)
noGelStimulated.drop(columns=dropCols, inplace=True)
gelRelaxed.drop(columns=dropCols, inplace=True)
gelStimulated.drop(columns=dropCols, inplace=True)

#EEG channels 
channels = ['EEG 1', ' EEG 2', ' EEG 3', ' EEG 4', ' EEG 5', ' EEG 6', ' EEG 7', ' EEG 8']


#extract EEG data 
#get only EEG channels as numpy arrays
noGelRelaxedEEG = noGelRelaxed[channels].values
noGelStimulatedEEG = noGelStimulated[channels].values
gelRelaxedEEG = gelRelaxed[channels].values
gelStimulatedEEG = gelStimulated[channels].values

print("Data loaded:")
print(f"No Gel Relaxed: {noGelRelaxedEEG.shape} ({noGelRelaxedEEG.shape[0]/samplingRate:.1f}s)")
print(f"No Gel Stimulated: {noGelStimulatedEEG.shape} ({noGelStimulatedEEG.shape[0]/samplingRate:.1f}s)")
print(f"Gel Relaxed: {gelRelaxedEEG.shape} ({gelRelaxedEEG.shape[0]/samplingRate:.1f}s)")
print(f"Gel Stimulated: {gelStimulatedEEG.shape} ({gelStimulatedEEG.shape[0]/samplingRate:.1f}s)")

#bandpass filter
def bandpass(data, lowcut=0.1, highcut=50):
    nyquist = samplingRate / 2
    b, a = signal.butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

noGelRelaxedFiltered = bandpass(noGelRelaxedEEG)
noGelStimulatedFiltered = bandpass(noGelStimulatedEEG)
gelRelaxedFiltered = bandpass(gelRelaxedEEG)
gelStimulatedFiltered = bandpass(gelStimulatedEEG)

#compute power spectral density
def computePSD(data):
    freqs, psd = signal.welch(data, fs=samplingRate, nperseg=samplingRate*2, axis=0)
    return freqs, psd

#compute PSD for each condition
freqs, noGelRelaxedPSD = computePSD(noGelRelaxedFiltered)
_, noGelStimulatedPSD = computePSD(noGelStimulatedFiltered)
_, gelRelaxedPSD = computePSD(gelRelaxedFiltered)
_, gelStimulatedPSD = computePSD(gelStimulatedFiltered)

#compute band power
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 50)
}

def getBandPower(freqs, psd, lowFreq, highFreq):
    idx = (freqs >= lowFreq) & (freqs <= highFreq)
    return np.mean(psd[idx, :], axis=0)

#calculate band powers for each condition
noGelRelaxedBands = {}
noGelStimulatedBands = {}
gelRelaxedBands = {}
gelStimulatedBands = {}

for bandName, (low, high) in bands.items():
    noGelRelaxedBands[bandName] = getBandPower(freqs, noGelRelaxedPSD, low, high)
    noGelStimulatedBands[bandName] = getBandPower(freqs, noGelStimulatedPSD, low, high)
    gelRelaxedBands[bandName] = getBandPower(freqs, gelRelaxedPSD, low, high)
    gelStimulatedBands[bandName] = getBandPower(freqs, gelStimulatedPSD, low, high)


#1. plot averaged PSD across all channels
plt.figure(figsize=(12, 6))

# average across all channels
noGelRelaxedPSD_avg = np.mean(noGelRelaxedPSD, axis=1)
noGelStimulatedPSD_avg = np.mean(noGelStimulatedPSD, axis=1)
gelRelaxedPSD_avg = np.mean(gelRelaxedPSD, axis=1)
gelStimulatedPSD_avg = np.mean(gelStimulatedPSD, axis=1)

plt.semilogy(freqs[freqs <= 50], noGelRelaxedPSD_avg[freqs <= 50], 
            label='No Gel Relaxed', linewidth=2, color='#3498db')
plt.semilogy(freqs[freqs <= 50], noGelStimulatedPSD_avg[freqs <= 50], 
            label='No Gel Stimulated', linewidth=2, color='#e74c3c')
plt.semilogy(freqs[freqs <= 50], gelRelaxedPSD_avg[freqs <= 50], 
            label='Gel Relaxed', linewidth=2, color='#2ecc71')
plt.semilogy(freqs[freqs <= 50], gelStimulatedPSD_avg[freqs <= 50], 
            label='Gel Stimulated', linewidth=2, color='#f39c12')

plt.axvspan(8, 13, alpha=0.2, color='green', label='Alpha Band')

plt.title('Power Spectral Density - Average Across All 8 Channels', fontsize=14, fontweight='bold')
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Power (μV²/Hz)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

#2. plot averaged band power comparison
bandNames = list(bands.keys())
x = np.arange(len(bandNames))
width = 0.2

plt.figure(figsize=(12, 6))

# average powers across all channels for each condition
noGelRelaxedPowers_avg = [np.mean(noGelRelaxedBands[band]) for band in bandNames]
noGelStimulatedPowers_avg = [np.mean(noGelStimulatedBands[band]) for band in bandNames]
gelRelaxedPowers_avg = [np.mean(gelRelaxedBands[band]) for band in bandNames]
gelStimulatedPowers_avg = [np.mean(gelStimulatedBands[band]) for band in bandNames]

# plot bars
plt.bar(x - 1.5*width, noGelRelaxedPowers_avg, width, label='No Gel Relaxed', alpha=0.8, color='#3498db')
plt.bar(x - 0.5*width, noGelStimulatedPowers_avg, width, label='No Gel Stimulated', alpha=0.8, color='#e74c3c')
plt.bar(x + 0.5*width, gelRelaxedPowers_avg, width, label='Gel Relaxed', alpha=0.8, color='#2ecc71')
plt.bar(x + 1.5*width, gelStimulatedPowers_avg, width, label='Gel Stimulated', alpha=0.8, color='#f39c12')

plt.title('Band Power Comparison - Average Across All 8 Channels', fontsize=14, fontweight='bold')
plt.xlabel('Frequency Band', fontsize=12)
plt.ylabel('Power (μV²/Hz)', fontsize=12)
plt.xticks(x, bandNames)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

#3. plot average alpha power across all channels
plt.figure(figsize=(10, 6))

# average alpha power across all channels for each condition
noGelRelaxedAlpha = np.mean(noGelRelaxedBands['Alpha'])
noGelStimulatedAlpha = np.mean(noGelStimulatedBands['Alpha'])
gelRelaxedAlpha = np.mean(gelRelaxedBands['Alpha'])
gelStimulatedAlpha = np.mean(gelStimulatedBands['Alpha'])

conditions = ['No Gel\nRelaxed', 'No Gel\nStimulated', 'Gel\nRelaxed', 'Gel\nStimulated']
alphaPowers = [noGelRelaxedAlpha, noGelStimulatedAlpha, gelRelaxedAlpha, gelStimulatedAlpha]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

bars = plt.bar(conditions, alphaPowers, color=colors, alpha=0.8)
plt.title('Average Alpha Power (8-13 Hz) Across All Channels', fontsize=14, fontweight='bold')
plt.ylabel('Alpha Power (μV²/Hz)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()

#4. plot gel effect
plt.figure(figsize=(10, 6))

# calculate gel effect for each band (average across channels)
gelEffectRelaxed = []
gelEffectStimulated = []

for band in bandNames:
    noGelRelPower = np.mean(noGelRelaxedBands[band])
    gelRelPower = np.mean(gelRelaxedBands[band])
    noGelStimPower = np.mean(noGelStimulatedBands[band])
    gelStimPower = np.mean(gelStimulatedBands[band])
    
    changeRelaxed = (gelRelPower - noGelRelPower) / noGelRelPower * 100
    changeStimulated = (gelStimPower - noGelStimPower) / noGelStimPower * 100
    
    gelEffectRelaxed.append(changeRelaxed)
    gelEffectStimulated.append(changeStimulated)

x = np.arange(len(bandNames))
width = 0.35

plt.bar(x - width/2, gelEffectRelaxed, width, label='Relaxed', alpha=0.8, color='#2ecc71')
plt.bar(x + width/2, gelEffectStimulated, width, label='Stimulated', alpha=0.8, color='#f39c12')
plt.axhline(y=0, color='black', linewidth=0.5)

plt.title('Gel Effect (% Change in Power) - Average Across All Channels', fontsize=14, fontweight='bold')
plt.xlabel('Frequency Band', fontsize=12)
plt.ylabel('Change (%)', fontsize=12)
plt.xticks(x, bandNames)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

#5. plot stimulation effect
plt.figure(figsize=(10, 6))

# calculate stimulation effect for each band (average across channels)
stimEffectNoGel = []
stimEffectGel = []

for band in bandNames:
    noGelRelPower = np.mean(noGelRelaxedBands[band])
    noGelStimPower = np.mean(noGelStimulatedBands[band])
    gelRelPower = np.mean(gelRelaxedBands[band])
    gelStimPower = np.mean(gelStimulatedBands[band])
    
    changeNoGel = (noGelStimPower - noGelRelPower) / noGelRelPower * 100
    changeGel = (gelStimPower - gelRelPower) / gelRelPower * 100
    
    stimEffectNoGel.append(changeNoGel)
    stimEffectGel.append(changeGel)

x = np.arange(len(bandNames))
width = 0.35

plt.bar(x - width/2, stimEffectNoGel, width, label='No Gel', alpha=0.8, color='#e74c3c')
plt.bar(x + width/2, stimEffectGel, width, label='With Gel', alpha=0.8, color='#f39c12')
plt.axhline(y=0, color='black', linewidth=0.5)

plt.title('Stimulation Effect (% Change in Power) - Average Across All Channels', fontsize=14, fontweight='bold')
plt.xlabel('Frequency Band', fontsize=12)
plt.ylabel('Change (%)', fontsize=12)
plt.xticks(x, bandNames)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')


plt.tight_layout()


plt.show()