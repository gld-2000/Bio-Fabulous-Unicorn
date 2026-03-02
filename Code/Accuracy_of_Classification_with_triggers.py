import mne
import numpy as np
from sklearn.cross_decomposition import CCA


fs = 250
window_length = 10 #Full trial duration (seconds)
freqs = [6, 7.5, 10, 15]
harmonics = 2

window_sec = 2
step_sec = 0.5

event_id = {
    'upper':  1,   # 15 Hz
    'right':  2,   # 6 Hz
    'lower':  3,   # 10 Hz
    'left':   4,   # 7.5 Hz
}

target_freqs = {
    'upper': 15,
    'right': 6,
    'lower': 10,
    'left': 7.5
}

def generate_reference(freq, fs, duration, harmonics=2):
    t = np.arange(0, duration, 1/fs)
    ref = []
    for h in range(1, harmonics+1):
        ref.append(np.sin(2*np.pi*freq*h*t))
        ref.append(np.cos(2*np.pi*freq*h*t))
    return np.array(ref).T  # (samples, 2*harmonics)


def detect_ssvep(eeg_window, fs):

    scores = []
    window_duration = eeg_window.shape[1] / fs

    for f in freqs:
        ref = generate_reference(f, fs, window_duration, harmonics)
        cca = CCA(n_components=1)

        cca.fit(eeg_window.T, ref)
        U, V = cca.transform(eeg_window.T, ref)

        corr = np.corrcoef(U.T, V.T)[0, 1]
        scores.append(corr)

    return freqs[np.argmax(scores)]
#sklearn.cross_decomposition.CCA expects input shape like this:(n_samples, n_features(channels here))


def sliding_window_trial_accuracy(trial_data, true_freq, fs,window_sec=2, step_sec=0.5):

    n_samples = trial_data.shape[1] #shape[1]=number of samples=2500
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    correct = 0
    total = 0
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        window = trial_data[:, start:end] 

        detected_freq = detect_ssvep(window, fs)
        if detected_freq == true_freq:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy, correct, total


raw = mne.io.read_raw_bdf(
    r"F:\lea\back to study\2nd semester\Biofablab\Git codes\Recordings\Flicker recordings\Wet_OSCAR-\UnicornRecorder_17_02_2026_17_04_15.bdf",
    preload=True
)


events = mne.find_events(raw, stim_channel='Status')

print("\nTotal triggers found:", len(events))
unique, counts = np.unique(events[:, 2], return_counts=True)
print("Trigger counts:")
for u, c in zip(unique, counts):
    print(f"Trigger {u}: {c} times")

raw.pick('eeg')
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
raw.pick(['Pz', 'PO7', 'Oz', 'PO8'])


epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=0,
    tmax=window_length,
    baseline=None,
    preload=True
)

print("\nNumber of trials per condition:")
for label in event_id.keys():
    print(label, ":", len(epochs[label]))

#Sliding window classification per trial
print("\nRunning Sliding Window CCA per trial...\n")
trial_accuracies = []
for label in event_id.keys():
    true_freq = target_freqs[label]
    trials = epochs[label].get_data()

    print(f"\nCondition: {label} (True freq: {true_freq} Hz)")
    for i, trial in enumerate(trials):
        acc, correct, total = sliding_window_trial_accuracy(
            trial,
            true_freq,
            fs,
            window_sec=window_sec,
            step_sec=step_sec
        )

        trial_accuracies.append(acc)

        print(f"     Correct windows: {correct}")
        print(f"     Total windows:   {total}")
        print(f"     Trial accuracy:  {acc:.3f}")

#Overall accuracy
if len(trial_accuracies) > 0:
    overall_accuracy = np.mean(trial_accuracies)
    print("\nAverage Trial Accuracy:", overall_accuracy)
else:
    print("\nNo trials found.")