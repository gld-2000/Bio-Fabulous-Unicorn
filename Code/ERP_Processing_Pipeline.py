import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

samplingRate = 250  #Data is sampled at 250 Hz
windowSize = 0.5    #Number of seconds to observe post-trigger

noGel = pd.read_csv('./Recordings/No Gel/Auditory Cue Spike/UnicornRecorder_04_02_2026_15_27_030.csv')
gel = pd.read_csv('./Recordings/With Gel/Auditory Cue Spike/UnicornRecorder_04_02_2026_16_10_300.csv')


gel.drop(columns=[" ACC X", " ACC Y", " ACC Z", " GYR X", " GYR Y", " GYR Z", " BAT"], inplace=True)
noGel.drop(columns=[" ACC X", " ACC Y", " ACC Z", " GYR X", " GYR Y", " GYR Z", " BAT"], inplace=True)

#There should be 8 stimulation events
gelIndices = np.where(gel[' TRIG'].values == 4)
noGelIndices = np.where(noGel[' TRIG'].values == 4)
numGelStimulations = len(gelIndices[0])
numNoGelStimulations = len(noGelIndices[0])

channels = ['EEG 1', ' EEG 2', ' EEG 3', ' EEG 4', ' EEG 5', ' EEG 6', ' EEG 7', ' EEG 8']

#Shape will be [stimulationEvent][channel][dataPoints]
gelEEG = np.zeros(shape=(8, len(channels), int(samplingRate*windowSize)))
noGelEEG = np.zeros(shape=(8, len(channels), int(samplingRate*windowSize)))

print(numGelStimulations)
print(numNoGelStimulations)

#j keeps track of the channel index, and k keeps track of the number of stimulation events.
#i is for the purpose of accessing the correct column in the raw data structure.
j = 0
for i in gel[channels] :
    for k in range(numGelStimulations) :
        gelEEG[k][j] = gel[i][gelIndices[0][k]:gelIndices[0][k]+int(samplingRate*windowSize)]
    j = j + 1

j = 0
for i in noGel[channels] :
    for k in range(numNoGelStimulations) :
        noGelEEG[k][j] = noGel[i][noGelIndices[0][k]:noGelIndices[0][k]+int(samplingRate*windowSize)]
    j = j + 1

# Will have shape of [channel][dataPoints]
gelEEGProcessed = np.mean(gelEEG, axis=0)
noGelEEGProcessed = np.mean(noGelEEG, axis=0)

xAxis = np.arange(0, windowSize, 1/samplingRate)

#Plot the waveform of each trial on a single channel (for each channel)
for j in range(len(channels)) :
    plt.figure()
    plt.title('Channel ' + str(j+1) + ' Gel')
    for i in range(numGelStimulations) :
        plt.plot(xAxis, gelEEG[i][j])
    plt.plot(xAxis, gelEEGProcessed[j], linewidth=3, label='Average', color='black')

for j in range(len(channels)) :
    plt.figure()
    plt.title('Channel ' + str(j+1) + ' No Gel')
    for i in range(numNoGelStimulations) :
        plt.plot(xAxis, noGelEEG[i][j])
    plt.plot(xAxis, noGelEEGProcessed[j], linewidth=3, label='Average', color='black')

#Plot the average waveform across all trials (for each channel)
plt.figure()
for i in range(len(channels)) :
    plt.plot(xAxis, gelEEGProcessed[i], label=channels[i])
plt.title('Average EEG Response Per Channel with Gel')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.legend()

plt.figure()
for i in range(len(channels)) :
    plt.plot(xAxis, noGelEEGProcessed[i], label=channels[i])
plt.title('Average EEG Response Per Channel without Gel')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.legend()


# #Plot a single stimulation event for each channel to see the raw data
# plt.figure()
# for i in range(len(channels)) :
#     plt.plot(xAxis, gelEEG[0][i])

plt.show()
