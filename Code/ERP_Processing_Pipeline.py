import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SAMPLING_RATE = 250  #Data is sampled at 250 Hz
windowSize = 0.35    #Number of seconds to observe post-trigger
channels = ['EEG 1', ' EEG 2', ' EEG 3', ' EEG 4', ' EEG 5', ' EEG 6', ' EEG 7', ' EEG 8']

"""
Args:   signal - the raw data structure containing all the recorded data
        triggerIndex - the array indices of the stimulation events
        windowSize - the number of seconds to observe post-trigger
        baselineShiftMethod - the method to apply a baseline shift to the data (0 for none, 1 for 'mean', 2 for 'firstPoint')
Returns: gelEEG - a 3D array of shape [stimulationEvent][channel][dataPoints] containing the extracted data for each stimulation event and channel
"""
def extractDataPostTrigger(signal, triggerIndex, windowSize, baselineShiftMethod) :
    numTriggers = len(triggerIndex[0])
    windowedData = np.zeros(shape=(numTriggers, len(channels), int(SAMPLING_RATE*windowSize)))
    
    #j keeps track of the channel index, and k keeps track of the number of stimulation events.
    #i is for the purpose of accessing the correct column in the raw data structure.
    j = 0
    for i in signal[channels] :
        for k in range(numTriggers) :
            windowedData[k][j] = signal[i][triggerIndex[0][k]:triggerIndex[0][k]+int(SAMPLING_RATE*windowSize)]
            #Apply baseline shift
            if baselineShiftMethod == 1 :       # Mean
                windowedData[k][j] = windowedData[k][j] - np.mean(windowedData[k][j])
            elif baselineShiftMethod == 2 :     # First point
                windowedData[k][j] = windowedData[k][j] - windowedData[k][j][0]
        j = j + 1
    return windowedData

"""
Same as previous function, but extracts data from a window before the trigger instead of after the trigger.
"""
def extractDataPreTrigger(signal, triggerIndex, windowSize, baselineShiftMethod) :
    numTriggers = len(triggerIndex[0])
    windowedData = np.zeros(shape=(numTriggers, len(channels), int(SAMPLING_RATE*windowSize)))
    
    #j keeps track of the channel index, and k keeps track of the number of stimulation events.
    #i is for the purpose of accessing the correct column in the raw data structure.
    j = 0
    for i in signal[channels] :
        for k in range(numTriggers) :
            windowedData[k][j] = signal[i][triggerIndex[0][k]-int(SAMPLING_RATE*windowSize):triggerIndex[0][k]]
            #Apply baseline shift
            if baselineShiftMethod == 1 :       # Mean
                windowedData[k][j] = windowedData[k][j] - np.mean(windowedData[k][j])
            elif baselineShiftMethod == 2 :     # First point
                windowedData[k][j] = windowedData[k][j] - windowedData[k][j][0]
        j = j + 1
    return windowedData



## Main code starts here

#Load CSV data
noGel = pd.read_csv('./Recordings/No Gel/Auditory Cue Spike/UnicornRecorder_04_02_2026_15_27_030.csv')
gel = pd.read_csv('./Recordings/With Gel/Auditory Cue Spike/UnicornRecorder_04_02_2026_16_10_300.csv')

#Drop unwanted channels
gel.drop(columns=[" ACC X", " ACC Y", " ACC Z", " GYR X", " GYR Y", " GYR Z", " BAT"], inplace=True)
noGel.drop(columns=[" ACC X", " ACC Y", " ACC Z", " GYR X", " GYR Y", " GYR Z", " BAT"], inplace=True)

#Extract the array indices for each stimulation event
gelCues = np.where(gel[' TRIG'].values == 1)
noGelCues = np.where(noGel[' TRIG'].values == 1)
gelResponses = np.where(gel[' TRIG'].values == 4)
noGelResponses = np.where(noGel[' TRIG'].values == 4)

#Calculate reponse time for fun :D
gelResponseTime = (gelResponses[0]-gelCues[0])*(1/SAMPLING_RATE)
noGelResponseTime = (noGelResponses[0]-noGelCues[0])*(1/SAMPLING_RATE)



# Extract data of interest (shape will be [stimulationEvent][channel][dataPoints])
gelEEG = extractDataPostTrigger(gel, gelCues, windowSize, baselineShiftMethod=1)
noGelEEG = extractDataPostTrigger(noGel, noGelCues, windowSize, baselineShiftMethod=1)

# Will have shape of [channel][dataPoints]
gelEEGProcessed = np.mean(gelEEG, axis=0)
noGelEEGProcessed = np.mean(noGelEEG, axis=0)



## Plot Data

xAxis = np.linspace(0, windowSize, int(SAMPLING_RATE*windowSize))
numGelStimulations = len(gelCues[0])
numNoGelStimulations = len(noGelCues[0])

#Plot the waveform of each trial on a single channel (for each channel)
for j in range(len(channels)) :
    plt.figure()
    plt.title('Channel ' + str(j+1) + ' Gel')
    for i in range(numGelStimulations) :
        plt.plot(xAxis, gelEEG[i][j])
    plt.plot(xAxis, gelEEGProcessed[j], linewidth=3, label='Average', color='black')
    plt.gca().set_ylim([-60,60])

for j in range(len(channels)) :
    plt.figure()
    plt.title('Channel ' + str(j+1) + ' No Gel')
    for i in range(numNoGelStimulations) :
        plt.plot(xAxis, noGelEEG[i][j])
    plt.plot(xAxis, noGelEEGProcessed[j], linewidth=3, label='Average', color='black')
    plt.gca().set_ylim([-60,60])



#Plot the average waveform across all trials (for each channel)
plt.figure()
for i in range(len(channels)) :
    plt.plot(xAxis, gelEEGProcessed[i], label=channels[i])
plt.title('Average EEG Response Per Channel with Gel')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.legend(loc='lower right')
plt.gca().set_ylim([-30,25])

plt.figure()
for i in range(len(channels)) :
    plt.plot(xAxis, noGelEEGProcessed[i], label=channels[i])
plt.title('Average EEG Response Per Channel without Gel')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.legend(loc='lower right')
plt.gca().set_ylim([-30,25])


# #Plot a single stimulation event for each channel to see the raw data
# plt.figure()
# for i in range(len(channels)) :
#     plt.plot(xAxis, gelEEG[0][i])
# plt.gca().set_ylim([-40,40])

plt.show()
