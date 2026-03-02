import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

bdf_path = r"C:\Users\NTC\Documents\Bio-Fabulous-Unicorn\Recordings\Flicker recordings\Sin_Dry_Neutral\UnicornRecorder_25_02_2026_15_18_46.bdf"

raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)

# --- events from Status (do this before dropping Status) ---
events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose=True)
print("events (first 20):\n", events[:20])
print("unique event codes:", sorted(set(events[:, 2])))
print("event times (s) first 20:", events[:20, 0] / raw.info["sfreq"])

# --- keep EEG only ---
raw.pick("eeg")

# cleanup
raw.notch_filter([50], verbose=False)
raw.filter(l_freq=0.1, h_freq=None, verbose=False)
raw.set_eeg_reference("average", verbose=False)

# trigger code -> condition mapping
event_id = {
    "13.5hz_up":   1,
    "8.5hz_right": 2,
    "12hz_down":   3,
    "10hz_left":   4,
}

# epoch 0..10s after each cue
epochs = mne.Epochs(raw, events, event_id=event_id,
                    tmin=0.0, tmax=10.0,
                    baseline=None, preload=True, verbose=False)

# PSD like before (analyze 1..10s)
tmin, tmax = 1.0, 10.0
sfreq = epochs.info["sfreq"]

spectrum = epochs.compute_psd(
    method="welch",
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    tmin=tmin, tmax=tmax,
    fmin=1, fmax=40,
    window="boxcar",
    verbose=False,
)

psds, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, n_ch, n_freq)

# targets + harmonics
targets = [8.5, 10.0, 12.0, 13.5]
harmonics = [17.0, 20.0, 24.0, 27.0]

# print per-condition peak values
for cond in epochs.event_id.keys():
    p = spectrum[cond].get_data().mean(axis=(0, 1))
    print("\n", cond)
    for f in targets + harmonics:
        idx = np.argmin(np.abs(freqs - f))
        print(f"  {freqs[idx]:5.2f} Hz : {p[idx]:.3e}")

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

snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)

# --- Per-condition PSD + SNR plots (8 panels: PSD and SNR for each direction) ---

fmin, fmax = 1, 40
mask = (freqs >= fmin) & (freqs <= fmax)

# fundamentals + harmonics, color-coded
freq_colors = {
    13.5: "tab:blue",    # up
    8.5:  "tab:red",     # right
    12.0: "tab:orange",  # down
    10.0: "tab:green",   # left
}
harmonics_map = {
    13.5: [27.0],
    12.0: [24.0],
    10.0: [20.0],
    8.5:  [17.0],
}

# Map condition name -> the event code (from your event_id)
conds = [
    ("13.5hz_up",    1),
    ("8.5hz_right",  2),
    ("12hz_down",    3),
    ("10hz_left",    4),
]

# We'll use epochs.events[:,2] to pick trials by event code
event_codes = epochs.events[:, 2]

fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(10, 10))

for row, (cond_name, code) in enumerate(conds):
    idx = np.where(event_codes == code)[0]
    if len(idx) == 0:
        # No epochs of this type; leave blank but label it
        axes[row, 0].set_title(f"{cond_name} (no epochs)")
        axes[row, 1].set_title(f"{cond_name} (no epochs)")
        continue

    # Select PSD/SNR for only this condition: shapes (n_ep, n_ch, n_freq)
    psd_c = psds[idx, :, :]
    snr_c = snrs[idx, :, :]

    # PSD: convert to dB first, then mean/std across epochs+channels
    psd_db = 10 * np.log10(psd_c)
    psd_mean = psd_db.mean(axis=(0, 1))[mask]
    psd_std  = psd_db.std(axis=(0, 1))[mask]

    ax_psd = axes[row, 0]
    ax_psd.plot(freqs[mask], psd_mean)
    ax_psd.fill_between(freqs[mask], psd_mean - psd_std, psd_mean + psd_std, alpha=0.2)
    ax_psd.set_ylabel(cond_name.replace("_", "\n"))
    if row == 0:
        ax_psd.set_title("PSD [dB]")
    ax_psd.set_xlim(fmin, fmax)

    # SNR: use nanmean/nanstd because edges are NaN
    snr_mean = np.nanmean(snr_c, axis=(0, 1))[mask]
    snr_std  = np.nanstd(snr_c, axis=(0, 1))[mask]

    ax_snr = axes[row, 1]
    ax_snr.plot(freqs[mask], snr_mean)
    ax_snr.fill_between(freqs[mask], snr_mean - snr_std, snr_mean + snr_std, alpha=0.2)
    if row == 0:
        ax_snr.set_title("SNR")
    ax_snr.set_xlim(fmin, fmax)

    # Add the colored frequency markers to both columns
    for ax in (ax_psd, ax_snr):
        for f0, c in freq_colors.items():
            ax.axvline(f0, color=c, linestyle="-", linewidth=1.5, alpha=0.8)
        for f0, hs in harmonics_map.items():
            c = freq_colors[f0]
            for h in hs:
                ax.axvline(h, color=c, linestyle="--", linewidth=1.2, alpha=0.6)

# X label only on bottom row
axes[-1, 0].set_xlabel("Frequency [Hz]")
axes[-1, 1].set_xlabel("Frequency [Hz]")

# --- legend for vertical lines (fundamentals + harmonics) ---
legend_handles = [
    Line2D([0], [0], color="tab:blue",   lw=1.8, linestyle="-",  label="Up 13.5 Hz"),
    Line2D([0], [0], color="tab:red",    lw=1.8, linestyle="-",  label="Right 8.5 Hz"),
    Line2D([0], [0], color="tab:orange", lw=1.8, linestyle="-",  label="Down 12 Hz"),
    Line2D([0], [0], color="tab:green",  lw=1.8, linestyle="-",  label="Left 10 Hz"),
    Line2D([0], [0], color="black",      lw=1.2, linestyle="--", label="2nd harmonic (2f)"),
]

# Put one shared legend for the whole figure
fig.legend(handles=legend_handles, loc="upper center", ncol=5, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show(block=False)
input("Press Enter to close...")