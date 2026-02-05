import mne
import pandas as pd
import numpy as np

noGel = pd.read_csv('./Recordings/No Gel/Auditory Cue Spike/UnicornRecorder_04_02_2026_15_27_030.csv')
gel = pd.read_csv('./Recordings/With Gel/Auditory Cue Spike/UnicornRecorder_04_02_2026_16_10_300.csv')


gel.drop(columns=[" ACC X", " ACC Y", " ACC Z", " GYR X", " GYR Y", " GYR Z", " BAT"], inplace=True)
noGel.drop(columns=[" ACC X", " ACC Y", " ACC Z", " GYR X", " GYR Y", " GYR Z", " BAT"], inplace=True)

indices = np.where(gel[' TRIG'].values == 4)

print(indices[0][0])
