import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

fs = 250
window_length = 10  

freqs = [8.5, 10, 12, 13.5]
harmonics = 2

window_sec = 2
step_sec = 0.5

event_id = {
    'upper':  1,   
    'right':  2,   
    'lower':  3,   
    'left':   4,   
}

target_freqs = {
    'upper': 13.5,
    'right': 8.5,
    'lower': 12,
    'left': 10
}


def generate_reference(freq, fs, duration, harmonics=2):

    t = np.arange(0, duration, 1/fs)
    ref = []

    for h in range(1, harmonics + 1):
        ref.append(np.sin(2 * np.pi * freq * h * t))
        ref.append(np.cos(2 * np.pi * freq * h * t))

    return np.array(ref).T  #shape: (samples, 2*harmonics)


def compute_cca_scores(eeg_window, fs):

    scores = []
    window_duration = eeg_window.shape[1] / fs

    for f in freqs:

        ref = generate_reference(f, fs, window_duration, harmonics)

        cca = CCA(n_components=1)

        cca.fit(eeg_window.T, ref)
        U, V = cca.transform(eeg_window.T, ref)

        corr = np.corrcoef(U.T, V.T)[0, 1]
        scores.append(corr)

    return scores  #list of 4 correlations

#To collect correlation scores for one trial
def collect_trial_scores(trial_data, fs, window_sec=2, step_sec=0.5):
    n_samples = trial_data.shape[1]
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    freq_scores = {f: [] for f in freqs}

    for start in range(0, n_samples - window_samples + 1, step_samples):

        end = start + window_samples
        window = trial_data[:, start:end]

        scores = compute_cca_scores(window, fs)

        for i, f in enumerate(freqs):
            freq_scores[f].append(scores[i])

    return freq_scores


raw = mne.io.read_raw_bdf(
    r"F:\lea\back to study\2nd semester\Biofablab\Git codes\Recordings\Flicker recordings\Sin_Dry_OSCAR+\UnicornRecorder_25_02_2026_15_11_24.bdf",
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

#Plot histograms per trial
print("\nGenerating Histogram Plots...\n")
for label in event_id.keys():
    trials = epochs[label].get_data()
    if len(trials) == 0:
        continue

    trial = trials[0]

    freq_scores = collect_trial_scores(
        trial,
        fs,
        window_sec=window_sec,
        step_sec=step_sec
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"CCA Correlation Histograms - {label}")

    axes = axes.flatten()

    for i, f in enumerate(freqs):

        axes[i].hist(freq_scores[f], bins=15)
        axes[i].set_title(f"{f} Hz")
        axes[i].set_xlabel("Correlation")
        axes[i].set_ylabel("Count")
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 5)  # set y-limit based on data

    plt.tight_layout()
    plt.show()

