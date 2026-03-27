import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import scipy


#Data analysis variables
numHarmonics = 2    #The number of harmonics to consider for the CCA analysis
freqsOfInterest = np.array([5.5,6,6.5,7])  #SSVEP frequencies of interest

trialDuration = 10.0

windowSizes = np.arange(0.5, 4.0, 0.25) #seconds - List of window sizes to analyze (from 0.5s to 4s in increments of 0.25s)

stepSize = 0.5

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

epochData_P0 = []
epochData_P5 = []

#Full frequencies
bdf_path = np.array([r".\Recordings\Flicker recordings\SinPanel\p0\5.3.GavinUnicornRawDataRecorder_05_03_2026_15_54_56.bdf",
                     #r".\Recordings\Flicker recordings\SinPanel\p0\12.3.HelenaUnicornRawDataRecorder_12_03_2026_17_47_44.bdf", #Excluded to match the number of P5 and P0 files
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


    #Gather epoched data for CCA analysis later
    tempEpochData = []
    for i in epochs.event_id.keys():
        tempEpochData.append(epochs.get_data(item=i))
    #Handle p0 files
    if numSubject < numFullFrequencyFiles:
        if numSubject == 0:
            epochData_P0 = np.array(tempEpochData)
        else:
            epochData_P0 = np.append(epochData_P0, tempEpochData, axis=1)
    #Handle p5 files
    else:
        if numSubject == numFullFrequencyFiles:
            epochData_P5 = np.array(tempEpochData)
        else:
            epochData_P5 = np.append(epochData_P5, tempEpochData, axis=1)
    print("///////////////////////////")

#Add to a single variable (shape = n_frequencies, n_trials, n_channels, n_samples)
epochData = np.empty((np.size(epochData_P0, axis=0) + np.size(epochData_P5, axis=0), np.size(epochData_P0, axis=1), np.size(epochData_P0, axis=2), np.size(epochData_P0, axis=3)))
epochData[0,:,:,:] = epochData_P5[0,:,:,:]
epochData[1,:,:,:] = epochData_P0[0,:,:,:]
epochData[2,:,:,:] = epochData_P5[1,:,:,:]
epochData[3,:,:,:] = epochData_P0[1,:,:,:]

numTrialTypes = np.size(epochData, axis=0)
numTrialsPerCond = np.size(epochData, axis=1)

print("epochData Shape:", np.shape(epochData))

#Create an array to store the number of correct classifications for each window size and event
#Shape is [trialType, windowSize]
numCorrect   = np.zeros((numTrialTypes, len(windowSizes)), dtype=int)    #Number of correct classifications
totalWindows = np.zeros((numTrialTypes, len(windowSizes)), dtype=int)    #Total number of classification performed
accuracy     = np.empty((numTrialTypes, len(windowSizes)), dtype=float)  #Accuracy of the classifications (numCorrect / totalWindows * 100)
#Shape is [trialType, windowSize, Frequency][Scores (numTrials * numWindows)]
ccaScores    = np.empty((numTrialTypes, len(windowSizes), len(freqsOfInterest)), dtype=list)       #Jagged list containing the CCA scores for every classificaiton that is performed
diffOfScore  = np.empty((numTrialTypes, len(windowSizes)), dtype=list)
#Shape is [trialType, windowSize, Frequency]
ccaAvgScore  = np.empty((numTrialTypes, len(windowSizes), len(freqsOfInterest)), dtype=float)   #Mean CCA score for each stimulation condition and window size
ccaVariance  = np.empty((numTrialTypes, len(windowSizes), len(freqsOfInterest)), dtype=float)    #Variance of CCA scores for each stimulation condition and window size
avgDiffScore = np.empty((numTrialTypes, len(windowSizes)), dtype=float)

#Analyze data using CCA
#Analyze different window sizes
for numWindowSize, windowSize in enumerate(windowSizes):
    #Determine length of array
    dataLength = np.size(epochData, axis=3)
    #Generate a new reference for each window size
    referenceWaves = []
    scansPerWindow = int(np.ceil(windowSize * 250))
    for f in freqsOfInterest:
        referenceWaves.append(generate_reference(f, 250, np.ceil(windowSize*250) / 250, numHarmonics))
    #Analyze each trial type
    for trialType in range(numTrialTypes):
        numWindows = 0
        #Analyze each stimulation event
        for trialNum in range(numTrialsPerCond):
            #Analyze the trial sub-windows, discarding the first one
            for start in range(125, dataLength, scansPerWindow):
                end = start + int(np.ceil(windowSize * 250))
                if end < dataLength:    #Ensure we don't go out of bounds
                    scores = detect_ssvep(epochData[trialType,trialNum,:,start:end], referenceWaves, freqsOfInterest)
                    result = classifyFromScores(scores, freqsOfInterest)
                    #Find the highest score that is not the correct classification (in order to identify median difference)
                    tempCurrScore = scores[trialType]
                    if trialType == 0:
                        tempNextScore = np.max(scores[1:])
                    elif trialType == numTrialTypes-1:
                        tempNextScore = np.max(scores[:trialType])
                    else:
                        tempNextScore = max(np.max(scores[:trialType]), np.max(scores[trialType+1:]))
                    #Add CCA score to list
                    if ccaScores[trialType,numWindowSize,0] is None:
                        diffOfScore[trialType,numWindowSize] = [(tempCurrScore - tempNextScore)]
                        for f in range(len(freqsOfInterest)):
                            ccaScores[trialType,numWindowSize,f] = [scores[f]]
                    else:
                        diffOfScore[trialType,numWindowSize].append(tempCurrScore - tempNextScore)
                        for f in range(len(freqsOfInterest)):
                            ccaScores[trialType,numWindowSize,f].append(scores[f])
                    if result == freqsOfInterest[trialType]:    #Check if the classification is correct
                        numCorrect[trialType,numWindowSize] += 1
                    numWindows += 1
        print("Window size:", windowSize, "| # of windows:", len(ccaScores[trialType,numWindowSize,0]), "| Should match:", numWindows)
        totalWindows[trialType,numWindowSize] = numWindows
        accuracy[trialType,numWindowSize] = numCorrect[trialType,numWindowSize] / totalWindows[trialType,numWindowSize] * 100
        for f in range(len(freqsOfInterest)):
            ccaAvgScore[trialType,numWindowSize,f] = np.median(ccaScores[trialType,numWindowSize,f][:])
            ccaVariance[trialType,numWindowSize,f] = np.var(ccaScores[trialType,numWindowSize,f][:])
            avgDiffScore[trialType,numWindowSize]  = np.median(diffOfScore[trialType,numWindowSize][:])

print("CCAScores shape:", np.shape(ccaScores))

##Display results

#Plot classification accuracy vs window size for each event
plt.figure()
for trialType in range(numTrialTypes):
    plt.plot(windowSizes, accuracy[trialType,:], label=(str(freqsOfInterest[trialType]) + " Hz"))
plt.xlabel("Window Size (s)")
plt.ylabel("Classification Accuracy (%)")
plt.title("Proportion of Correct Classifications per Stimulation Type")
plt.gca().set_ylim(0, 105)
plt.legend()


#Plot average score vs window size for each frequency in each event
fig, axs = plt.subplots(2,2)
for i in range(numTrialTypes):
    for f in range(len(freqsOfInterest)):
        axs[i//2,i%2].plot(windowSizes, ccaAvgScore[i,:,f], label=(str(freqsOfInterest[f]) + " Hz"))
    axs[i//2,i%2].set_title("Stim: " + str(freqsOfInterest[i]) + " Hz")
    axs[i//2,i%2].legend()
    axs[i//2,i%2].set_xlabel("Window Size (s)")
    axs[i//2,i%2].set_ylabel("Median Score")
fig.suptitle("Median CCA Scores")

#Plot score variance vs window size for each frequency in each event
fig, axs = plt.subplots(2,2)
for i in range(numTrialTypes):
    for f in range(len(freqsOfInterest)):
        axs[i//2,i%2].plot(windowSizes, ccaVariance[i,:,f], label=(str(freqsOfInterest[f]) + " Hz"))
    axs[i//2,i%2].set_title("Stim: " + str(freqsOfInterest[i]) + " Hz")
    axs[i//2,i%2].legend()
    axs[i//2,i%2].set_xlabel("Window Size (s)")
    axs[i//2,i%2].set_ylabel("Variance")
fig.suptitle("Variance of CCA Scores")

diffOfAvgScore = np.empty((numTrialTypes, len(windowSizes)))
#Find the difference between the highest and second-highest scores
for i in range(numTrialTypes):
    for windowSize in range(len(windowSizes)):
        currentVal = ccaAvgScore[i,windowSize,i]
        if i == 0:
            otherMaxVal = np.max(ccaAvgScore[i,windowSize,1:])
        elif i == numTrialTypes-1:
            otherMaxVal = np.max(ccaAvgScore[i,windowSize,:i])
        else:
            otherMaxVal = max(np.max(ccaAvgScore[i,windowSize,:i]), np.max(ccaAvgScore[i,windowSize,i+1:]))
        diffOfAvgScore[i,windowSize] = currentVal - otherMaxVal
        
#Plot average score vs window size for each frequency in each event
fig, axs = plt.subplots(2,2)
for i in range(numTrialTypes):
    axs[i//2,i%2].plot(windowSizes, diffOfAvgScore[i,:], label=(str(freqsOfInterest[i]) + " Hz"))
    axs[i//2,i%2].set_title("Stim: " + str(freqsOfInterest[i]) + " Hz")
    axs[i//2,i%2].legend()
    axs[i//2,i%2].set_xlabel("Window Size (s)")
    axs[i//2,i%2].set_ylabel("Difference in Median")
fig.suptitle("Difference of Median of CCA Scores (#1-#2)")

#Plot average score vs window size for each frequency in each event
fig, axs = plt.subplots(2,2)
for i in range(numTrialTypes):
    axs[i//2,i%2].plot(windowSizes, avgDiffScore[i,:], label=(str(freqsOfInterest[i]) + " Hz"))
    axs[i//2,i%2].set_title("Stim: " + str(freqsOfInterest[i]) + " Hz")
    axs[i//2,i%2].legend()
    axs[i//2,i%2].set_xlabel("Window Size (s)")
    axs[i//2,i%2].set_ylabel("Median Difference")
fig.suptitle("Median Difference of CCA Scores (#1-#2)")

plt.show()
