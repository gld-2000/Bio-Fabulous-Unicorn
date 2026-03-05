import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


stepSize = 0.5

#Frequencies as directions
FREQ_UP    = 13.5   #Hz
FREQ_DOWN  = 12     #Hz
FREQ_LEFT  = 10     #Hz
FREQ_RIGHT = 8.5    #Hz

#Data analysis variables
numHarmonics = 2    #The number of harmonics to consider for the CCA analysis
freqsOfInterest = [FREQ_UP, FREQ_RIGHT, FREQ_DOWN, FREQ_LEFT]  #SSVEP frequencies of interest

windowSizes = np.arange(0.5, 4.0, 0.25) #seconds - List of window sizes to analyze (from 0.5s to 4s in increments of 0.25s)

"""
Detect frequency using CCA

args: eeg_window - EEG data with shape (numChannels, numSamples)
    ref - Reference waveforms for CCA
    freqs - The frequencies of interest for classification

returns: A list of the scores for the given frequencies
"""
def detect_ssvep(eeg_window, ref, freqs):
    scores = np.empty(len(freqs), dtype=np.float64)
    for i in range(len(freqs)):
        cca = CCA(n_components=1)
        cca.fit(eeg_window.T, ref[i])      #Must transform since sklearn.cross_decomposition.CCA expects input shape like this:(n_samples, n_features(channels here))
        U, V = cca.transform(eeg_window.T, ref[i])
        corr = np.corrcoef(U.T, V.T)[0,1]
        scores[i] = corr
    return scores

def generate_reference(freq, fs, window_len, harmonics=2):
    t = np.arange(0, window_len, 1/fs) #time_vector from 0 to window_length with step size = 1 / sampling_rate
    ref = []
    for h in range(1, harmonics+1):
        ref.append(np.sin(2*np.pi*freq*h*t))
        ref.append(np.cos(2*np.pi*freq*h*t))
    return np.array(ref).T  #shape: (samples, 2*harmonics)

def classifyFromScores(scores, freqs):
    return freqs[np.argmax(scores)]


#Map anatomical names to hardware channels
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
channelsToKeep = ['Pz','PO7','Oz','PO8']


bdf_path = r".\Recordings\Flicker recordings\Sin_Dry_OSCAR+\UnicornRecorder_25_02_2026_15_11_24.bdf"

raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)

# --- events from Status (do this before dropping Status) ---
events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose=True)

# Isolate channels of interest
raw.rename_channels(mapping)
raw.pick(channelsToKeep)


# cleanup
raw.notch_filter([50], verbose=False)
raw.filter(l_freq=0.1, h_freq=None, verbose=False)
raw.set_eeg_reference("average", verbose=False)

refData = raw.get_data(start=11638, stop=11638+250)

# trigger code -> condition mapping
event_id = {
    "13.5hz_up":    1,
    "8.5hz_right":  2,
    "12hz_down":    3,
    "10hz_left":    4,
}


# epoch 0..10s after each cue
epochs = mne.Epochs(raw, events, event_id=event_id,
                    tmin=0.0, tmax=10.0,
                    baseline=None, preload=True, verbose=False)

#Takes shape of [Event][EventIndex, Channels, Samples]
epochData = []
for i in event_id.keys():
    epochData.append(epochs.get_data(item=i))

#Create an array to store the number of correct classifications for each window size and event
#Takes shape of [TrialInstance, TrialType, WindowSize]
numCorrect = np.zeros((np.size(events, axis=0)//len(event_id), len(event_id), len(windowSizes)), dtype=int)     #Number of correct classifications
totalWindows = np.empty((np.size(events, axis=0)//len(event_id), len(event_id), len(windowSizes)), dtype=int)   #Total number of classification performed
accuracy = np.empty((np.size(events, axis=0)//len(event_id), len(event_id), len(windowSizes)), dtype=float)       #Accuracy of classificaions (numCorrect / totalWindows * 100)
#Takes shape of [TrialType, WindowSize, Frequency][Scores]
ccaScores = np.empty((len(event_id), len(windowSizes), len(freqsOfInterest)), dtype=list)       #Jagged list containing the CCA scores for every classificaiton that was performed
#Takes shape of [TrialType, WindowSize, Frequency]
ccaMeanScore = np.empty((len(event_id), len(windowSizes), len(freqsOfInterest)), dtype=float)   #Mean CCA score for each stimulation condition and window size
ccaVariance = np.empty((len(event_id), len(windowSizes), len(freqsOfInterest)), dtype=float)    #Variance of CCA scores for each stimulation condition and window size

#Analyze data using CCA
#Analyze each trial
for i in range(len(event_id)):
    #Analyze different window sizes
    j = 0
    for windowSize in windowSizes: #seconds
        #stepSize = windowSize
        #Determine length of array
        (_,_,dataLength) = np.shape(epochData[i])
        #Generate a new reference for each window size
        referenceWaves = []
        scansPerWindow = int(np.ceil(windowSize * 250))
        for f in freqsOfInterest:
            referenceWaves.append(generate_reference(f, 250, np.ceil(windowSize*250) / 250, numHarmonics))
        k = 0
        #Analyze individual windows
        for start in range(0, dataLength, int(stepSize * 250)):
            if start + int(windowSize * 250) < dataLength: #Ensure we don't go out of bounds
                for trialNum in range(2):
                    scores = detect_ssvep(epochData[i][trialNum,:,start:start+int(np.ceil(windowSize*250))], referenceWaves, freqsOfInterest)
                    result = classifyFromScores(scores, freqsOfInterest)
                    #If the list has not been created, create it. Otherwise, extend the existing list
                    if ccaScores[i,j,0] is None:
                        ccaScores[i,j,0] = [scores[0]]
                        ccaScores[i,j,1] = [scores[1]]
                        ccaScores[i,j,2] = [scores[2]]
                        ccaScores[i,j,3] = [scores[3]]
                    else:
                        ccaScores[i,j,0].append(scores[0])
                        ccaScores[i,j,1].append(scores[1])
                        ccaScores[i,j,2].append(scores[2])
                        ccaScores[i,j,3].append(scores[3])
                    if result == freqsOfInterest[i]: #Check if the classification is correct
                        numCorrect[trialNum,i,j] += 1
                k += 1
        totalWindows[:,i,j] = k
        accuracy[:,i,j] = numCorrect[:,i,j]/totalWindows[:,i,j]*100
        for f in range(len(freqsOfInterest)):
            ccaMeanScore[i,j,f] = np.mean(ccaScores[i,j,f])
            ccaVariance[i,j,f] = np.var(ccaScores[i,j,f])
        j += 1




##Display results

#Print results
# for i, event in enumerate(event_id.keys()):
#     print("Frequency: ", event)
#     for j, windowSize in enumerate(windowSizes):
#         print("  Window Size: ", windowSize, "s - Accuracy: ", numCorrect[i,j], "/", totalWindows[i,j], " (", accuracy[i,j], "%)")

#Plot classification accuracy vs window size for each event
stimName = list(event_id.keys())
for i in range(len(event_id)):
    plt.figure()
    plt.plot(windowSizes, accuracy[0,i,:])
    plt.plot(windowSizes, accuracy[1,i,:])
    plt.xlabel("Window Size (s)")
    plt.ylabel("Classification Accuracy (%)")
    plt.gca().set_ylim(0, 105)
    plt.title("Stimulation: " + stimName[i])

#Plot average score vs window size for each frequency in each event
fig, axs = plt.subplots(2,2)
for i, event in enumerate(event_id.keys()):
    for f in range(len(freqsOfInterest)):
        axs[i//2,i%2].plot(windowSizes, ccaMeanScore[i,:,f])
        axs[i//2,i%2].set_title(event)
        axs[i//2,i%2].legend(event_id.keys())
        axs[i//2,i%2].set_xlabel("Window Size (s)")
        axs[i//2,i%2].set_ylabel("Mean Score")
fig.suptitle("Mean CCA Scores")

#Plot score variance vs window size for each frequency in each event
fig, axs = plt.subplots(2,2)
for i, event in enumerate(event_id.keys()):
    for f in range(len(freqsOfInterest)):
        axs[i//2,i%2].plot(windowSizes, ccaVariance[i,:,f])
        axs[i//2,i%2].set_title(event)
        axs[i//2,i%2].legend(event_id.keys())
        axs[i//2,i%2].set_xlabel("Window Size (s)")
        axs[i//2,i%2].set_ylabel("Variance")
fig.suptitle("Variance of CCA Scores")

#Modify to plot a single figure for presentation purposes
# plt.figure()
# for f in range(len(freqsOfInterest)):
#     plt.plot(windowSizes, ccaMeanScore[2,:,f])
#     plt.title(list(event_id.keys())[2])
#     plt.legend(event_id.keys())
#     axs = plt.gca()
#     axs.set_xlabel("Window Size (s)")
#     axs.set_ylabel("Mean Score")

plt.show()
