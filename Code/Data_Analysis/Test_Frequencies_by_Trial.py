import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import scipy


#Data analysis variables
numHarmonics = 2    #The number of harmonics to consider for the CCA analysis
freqsOfInterest = np.array([5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15])  #SSVEP frequencies of interest

trialDuration = 10.0

subWindowSize = 2.0
stepSize = 2.0

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


psds = []       #Takes shape of (n_frequencies, n_trials, n_ch, n_freq) in order of frequencies
freqs = []
epochData = []  #epochData takes shape of [Event][EventIndex, Channels, Samples]

#Full frequencies
bdf_path = np.array([r".\Recordings\Flicker recordings\SinPanel\p0\5.3.GavinUnicornRawDataRecorder_05_03_2026_15_54_56.bdf",
                     r".\Recordings\Flicker recordings\SinPanel\p0\12.3.HelenaUnicornRawDataRecorder_12_03_2026_17_47_44.bdf",
                     r".\Recordings\Flicker recordings\SinPanel\p0\12.3.HelenaUnicornRawDataRecorder_12_03_2026_18_27_32.bdf",
                     r".\Recordings\Flicker recordings\SinPanel\p0\13.3.MarceloUnicornRawDataRecorder_13_03_2026_13_44_34.bdf",
                     r".\Recordings\Flicker recordings\SinPanel\p0\13.3.MarceloUnicornRawDataRecorder_13_03_2026_14_12_43.bdf",
                     r".\Recordings\Flicker recordings\SinPanel\p0\13.3.MarceloUnicornRawDataRecorder_13_03_2026_16_29_33.bdf",
                     r".\Recordings\Flicker recordings\SinPanel\p0\18.3.GaelUnicornRawDataRecorder_18_03_2026_14_38_19.bdf",
                     r".\Recordings\Flicker recordings\SinPanel\p0\18.3.GavinUnicornRawDataRecorder_18_03_2026_15_45_00.bdf"
                     ],
                     dtype = np.str_)

#Half-frequencies
half_freqs_path = np.array([r".\Recordings\Flicker recordings\SinPanel\p5\5.3.GavinUnicornRawDataRecorder_05_03_2026_16_07_42.bdf",
                            r".\Recordings\Flicker recordings\SinPanel\p5\12.3.HelenaUnicornRawDataRecorder_12_03_2026_18_15_38.bdf",
                            r".\Recordings\Flicker recordings\SinPanel\p5\13.3.MarceloUnicornRawDataRecorder_13_03_2026_13_36_43.bdf",
                            r".\Recordings\Flicker recordings\SinPanel\p5\13.3.MarceloUnicornRawDataRecorder_13_03_2026_13_58_15.bdf",
                            r".\Recordings\Flicker recordings\SinPanel\p5\13.3.MarceloUnicornRawDataRecorder_13_03_2026_16_17_49.bdf",
                            r".\Recordings\Flicker recordings\SinPanel\p5\18.3.GaelUnicornRawDataRecorder_18_03_2026_14_49_09.bdf",
                            r".\Recordings\Flicker recordings\SinPanel\p5\18.3.GavinUnicornRawDataRecorder_18_03_2026_15_36_35.bdf"
                            ],
                            dtype = np.str_)

numFullFrequencyFiles = len(bdf_path)
print("Number of recordings for full frequencies:", numFullFrequencyFiles)
numHalfFrequencyFiles = len(half_freqs_path)
print("Number of recordings for half frequencies:", numHalfFrequencyFiles)

bdf_path = np.append(bdf_path, half_freqs_path)

#Import the data from each array
for numSubject, path in enumerate(bdf_path):
    print("File number: ", numSubject+1)
    raw = mne.io.read_raw_bdf(bdf_path[numSubject], preload=True, verbose=False)

    # --- events from Status (do this before dropping Status) ---
    events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose=True)

    # Isolate channels of interest
    raw.rename_channels(mapping)
    raw.pick(channelsToKeep)


    # cleanup
    raw.notch_filter([50], verbose=False)
    raw.filter(l_freq=0.1, h_freq=None, verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    # trigger code -> condition mapping
    trig_id = np.multiply(freqsOfInterest, 10).astype(int) #Triggers were 10x of each frequency


    # epoch 0..10s after each cue
    epochs = mne.Epochs(raw, events,
                        tmin=0.0, tmax=10.0,
                        baseline=None, preload=True, verbose=False)

    epochsToDelete = []
    keysToDelete = []
    #Drop indices of events that are not in the target frequencies
    for i in range(len(epochs)):
        #Record the epoch index if its trigger code doesn't correspond to the target frequencies
        if not (list(epochs[i].event_id.values())[0] in trig_id):
            epochsToDelete.append(i)
            keysToDelete.append(list(epochs[i].event_id.keys())[0])
            print("Dropping epoch with trigger code: ", list(epochs[i].event_id.keys())[0])
    epochs.drop(epochsToDelete)
    
    #Delete keys that are no longer present in the epochs structure
    keysToDelete = list(dict.fromkeys(keysToDelete)) #Remove duplicates by converting to a dict and then back to a list
    for i in range(len(keysToDelete)):
        del epochs.event_id[keysToDelete[i]]

    #PSD parameters
    tmin, tmax = 0.0, 10.0
    sfreq = epochs.info["sfreq"]

    #Compute PSD and return data
    spectrum = []
    for i in epochs.event_id.keys():
            spectrum.append(epochs[i].compute_psd(
                method="welch",
                n_fft=int(sfreq * (tmax - tmin)),
                n_overlap=0,
                tmin=tmin, tmax=tmax,
                fmin=1, fmax=50,
                window="boxcar",
                verbose=False,
            ))

    psdsTemp = []   #Will have shape (n_frequencies, n_trials, n_ch, n_freq) in order of frequencies
    freqsTemp = []
    for i in range(len(epochs.event_id.keys())):
        tempPSD, tempFreq = spectrum[i].get_data(return_freqs=True)
        psdsTemp.append(tempPSD)
        freqsTemp.append(tempFreq)
    if numSubject == 0:
        psds = np.array(psdsTemp)
        freqs = np.array(freqsTemp)
    else:
        psds = np.append(psds, psdsTemp, axis=1)
        #TODO: Fix freqs to track and plot the output of each new file, rather than just the first file
        #freqs = np.append(freqs, freqsTemp, axis=2) #Must append in a new direction since freqs is per trial type and not per trial instance

    #Gather epoched data for CCA analysis later
    tempEpochData = []
    for i in epochs.event_id.keys():
        tempEpochData.append(epochs.get_data(item=i))
    if numSubject == 0:
        epochData = np.array(tempEpochData)
    else:
        epochData = np.append(epochData, tempEpochData, axis=1)
    
    print("///////////////////////////")

print("psds shape:", np.shape(psds), "freqs shape:", np.shape(freqs))

# --- SNR computation ---
def snr_spectrum(psd, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1):
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"),
        axis=-1,
        arr=psd,
    )

    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)
    return psd / mean_noise

totalTrials = np.size(psds, axis=1)

snrs = []
for i in range(totalTrials):
    snrs.append(snr_spectrum(psds[:,i,:,:], noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1))
snrs = np.array(snrs)

print("snrs shape:", np.shape(snrs))

print("epochData Shape:", np.shape(epochData))

#Create an array to store the CCA scores for each frequency at each trial
numWindows =  (int(trialDuration)//int(subWindowSize))-1
#Takes shape of [TrialType, trialNum, windowNum, Scores]
ccaScores = np.empty((len(epochData), totalTrials, numWindows, len(freqsOfInterest)))

#Analyze data using CCA
#Analyze each trial
for i in range(len(epochData)):
    #Determine length of array
    (_,_,dataLength) = np.shape(epochData[i])
    #Generate a new reference for each window size
    referenceWaves = []
    scansPerWindow = int(np.ceil(subWindowSize * 250))
    for f in freqsOfInterest:
        referenceWaves.append(generate_reference(f, 250, np.ceil(subWindowSize*250) / 250, numHarmonics))
    for trialNum in range(totalTrials):
        #Analyze the trial sub-windows, discarding the first one
        for idx, start in enumerate(range(int(np.ceil(stepSize * 250)), dataLength, scansPerWindow)):
            end = start + int(np.ceil(subWindowSize * 250))
            if end < dataLength:
                scores = detect_ssvep(epochData[i,trialNum,:,start:end], referenceWaves, freqsOfInterest)
                ccaScores[i,trialNum,idx] = scores

print("CCAScores shape:", np.shape(ccaScores))

##Display results

#Plot average score vs window size for each frequency in each event
for fileSplit in (1,0):
    fig, axs = plt.subplots(5,2)
    for i, event in enumerate(freqsOfInterest[fileSplit::2]):  #For each frequency
        for windowNum in range(numWindows):
            if fileSplit == 1:
                for trialNum in range(numFullFrequencyFiles * 2):
                    axs[i%5,i//5].scatter(freqsOfInterest, ccaScores[i,trialNum,windowNum,:])
            else:
                for trialNum in range(numFullFrequencyFiles * 2, totalTrials):
                    axs[i%5,i//5].scatter(freqsOfInterest, ccaScores[i,trialNum,windowNum,:])
        axs[i%5,i//5].set_title("Frequency: " + str(event))
        axs[i%5,i//5].set_xlabel("Frequency (Hz)")
        axs[i%5,i//5].set_ylabel("Score")
        axs[i%5,i//5].set_ylim([0,0.9])
    fig.suptitle("CCA Scores")

#Create a new array that has the CCA scores stored in the order of frequencies
orderedCCAScores = np.empty((len(freqsOfInterest), (numFullFrequencyFiles + numHalfFrequencyFiles)*2*numWindows), )
idx = 0
for isWholeNum in (1,0):
    for i, event in enumerate(freqsOfInterest[isWholeNum::2]):  #For each frequency
        orderedCCAScores[i*2+isWholeNum, :] = np.ravel(ccaScores[i,:,:,i*2+isWholeNum])   #Reorder the CCA scores in order of 
print("Ordered CCA Scores shape:", np.shape(orderedCCAScores))

#Plot histograms of the CCA score that corresponds to the frequency of stimulation
for isWholeNum in (1,0):
    fig, axs = plt.subplots(5,2)
    for i, event in enumerate(freqsOfInterest[isWholeNum::2]):  #For each frequency
        axs[i%5,i//5].hist(np.ravel(ccaScores[i,:,:,(i*2+isWholeNum)]))  #Takes shape of [TrialType, trialNum][Scores]
        axs[i%5,i//5].set_title("Frequency: " + str(event))
        axs[i%5,i//5].set_xlabel("CCA Score")
        axs[i%5,i//5].set_ylabel("# of Trials")
        axs[i%5,i//5].set_xlim([0,0.9])
        axs[i%5,i//5].set_ylim([0,46])
    fig.suptitle("CCA Scores")

# upperBound = 31
# lowerBound = 5

# #Plot PSD channels for all trials
#TODO: Complete this to make it accurate
# for isWholeNum in (1,0):
#     for trialNum in range(totalTrials):
#         fig, axs = plt.subplots(5,2)
#         for i, event in enumerate(freqsOfInterest[isWholeNum::2]):  #For each frequency
#             for channelNum in range(np.size(snrs, axis=2)):
#                 axs[i%5,i//5].plot(freqs[i,:], snrs[trialNum,i,channelNum,:])
#             axs[i%5,i//5].set_title("Frequency: " + str(event))
#             axs[i%5,i//5].set_xlabel("Frequency (Hz)")
#             axs[i%5,i//5].set_ylabel("Relative Amp")
#             axs[i%5,i//5].set_ylim([-1.5,35])
#             axs[i%5,i//5].set_xlim([lowerBound,upperBound])
#         fig.suptitle("Power Spectral Density: Trial " + str(trialNum+1))

# #Plot PSD channels for both trials (averaging across channels for each trial)
# fig, axs = plt.subplots(5,2)
# for i, event in enumerate(epochs.event_id.keys()):  #For each frequency
#     for trialNum in range(totalTrials):
#         axs[i%5,i//5].plot(freqs[i,:], np.mean(snrs[trialNum,i,:,:], axis=0))
#     axs[i%5,i//5].set_title("Frequency: " + str((int(event)/10)))
#     axs[i%5,i//5].set_xlabel("Frequency (Hz)")
#     axs[i%5,i//5].set_ylabel("Mean Amplitude")
#     axs[i%5,i//5].set_ylim([-1.5,45])
#     axs[i%5,i//5].set_xlim([lowerBound,upperBound])
# fig.suptitle("Power Spectral Density (Channel Averaged)")

# #Plot score variance vs window size for each frequency in each event
# fig, axs = plt.subplots(5,2)
# for i, event in enumerate(epochs.event_id.keys()):
#     axs[i%5,i//5].scatter(freqsOfInterest, ccaVariance[i,:])
#     axs[i%5,i//5].set_title("Frequency: " + event)
#     axs[i%5,i//5].set_xlabel("Frequency (Hz)")
#     axs[i%5,i//5].set_ylabel("Variance")
# fig.suptitle("Variance of CCA Scores")

#Modify to plot a single figure for presentation purposes
# plt.figure()
# for f in range(len(freqsOfInterest)):
#     plt.plot(windowSizes, ccaMeanScore[2,:,f])
#     plt.title(list(epochs.event_id.keys())[2])
#     plt.legend(epochs.event_id.keys())
#     axs = plt.gca()
#     axs.set_xlabel("Window Size (s)")
#     axs.set_ylabel("Mean Score")

#Calculate CCA means
medianValues = np.median(orderedCCAScores, axis=1)

#Test for normality
normalityStats = scipy.stats.normaltest(orderedCCAScores,
                                        axis=1,
                                        nan_policy='raise')
#Test for similar variances
varianceStats = scipy.stats.levene(*[orderedCCAScores[i] for i in range(len(freqsOfInterest))],
                                   center='median',
                                   nan_policy='raise')

print("For the following tests, a p-value <0.05 indicates that the distribution is non-normal:")
for i, freq in enumerate(freqsOfInterest):
    print("Distribution for frequency:", freq, "| Median:", medianValues[i], "| Normality stat:", normalityStats.statistic[i],"| Normality p-value:", normalityStats.pvalue[i])
print("Variance stat:", varianceStats.statistic, "| Variance p-value (<0.05 means we should set ANOVA equal_var to False, True otherwise):", varianceStats.pvalue)

#Perform ANOVA
anovaFStat, anovaPStat = scipy.stats.f_oneway(*[orderedCCAScores[i] for i in range(len(freqsOfInterest))],
                                              equal_var = False,
                                              nan_policy = 'raise')
print("Welch's ANOVA analysis: F-Stat:", anovaFStat, "P-stat (<0.05 indicates significant difference between distributions):", anovaPStat)

largestMeanIndices = np.flip(np.argsort(medianValues)[-4:])    #Stores the indices of the 4 greatest means
print("Largest medians:", medianValues[largestMeanIndices])
print("Corresponding frequencies:", freqsOfInterest[largestMeanIndices])

plt.show()
