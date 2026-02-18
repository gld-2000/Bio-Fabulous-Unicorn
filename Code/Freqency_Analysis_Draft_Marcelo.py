import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

bdf_path = r"C:\Users\NTC\Documents\Bio-Fabulous-Unicorn\Recordings\Flicker recordings\Wet_OSCAR+\UnicornRawDataRecorder_17_02_2026_16_58_20.bdf"

raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=True)

#Events from Status
events = mne.find_events(mne.io.read_raw_bdf(bdf_path, preload=False, verbose=False), stim_channel="Status", shortest_event=1, verbose=False)

events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose=True)
print("events:", events)
print("event times (s):", events[:, 0] / raw.info["sfreq"])

raw.pick("eeg")

# Basic cleanup
raw.notch_filter([50], verbose=False)
raw.filter(l_freq=0.1, h_freq=None, verbose=False)
raw.set_eeg_reference("average", verbose=False)

event_id = {
    "15hz": 1, #up
    "10hz": 3, #down
    "7.5hz":  2, #right
    "6hz": 4, #left
}

epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.0, tmax=10.0, baseline=None, preload=True, verbose=False)

tmin, tmax = 1.0, 10.0
sfreq = epochs.info["sfreq"]
spectrum = epochs.compute_psd(method="welch", n_fft=int(sfreq * (tmax - tmin)), n_overlap=0, tmin=tmin, tmax=tmax, fmin=1, fmax=40, window="boxcar", verbose=False)

psds, freqs = spectrum.get_data(return_freqs=True)

#Print peak values near stimulus freqs for each condition
targets = [6, 7.5, 10, 15]
for cond in epochs.event_id.keys():
    p = spectrum[cond].get_data().mean(axis=(0, 1)) #avg epochs + channels
    print("\n", cond)
    for f in targets + [12, 20, 30]: #harmonics
        idx = np.argmin(np.abs(freqs - f))
        print(f"  {freqs[idx]:5.2f} Hz : {p[idx]:.3e}")

def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise

snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)

fmin, fmax = 1, 40

fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
freq_range = range(
    np.where(np.floor(freqs) == 1.0)[0][0], np.where(np.ceil(freqs) == fmax - 1)[0][0]
)

psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
psds_std = psds_plot.std(axis=(0, 1))[freq_range]
axes[0].plot(freqs[freq_range], psds_mean, color="b")
axes[0].fill_between(
    freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.2
)
axes[0].set(title="PSD spectrum", ylabel="Power Spectral Density [dB]")

# SNR spectrum
snr_mean = snrs.mean(axis=(0, 1))[freq_range]
snr_std = snrs.std(axis=(0, 1))[freq_range]

axes[1].plot(freqs[freq_range], snr_mean, color="r")
axes[1].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.2
)
axes[1].set(
    title="SNR spectrum",
    xlabel="Frequency [Hz]",
    ylabel="SNR",
    ylim=[-2, 30],
    xlim=[fmin, fmax],
)
fig.show()

input("Press Enter to close...")
