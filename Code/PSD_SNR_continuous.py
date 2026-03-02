import mne, numpy as np
import matplotlib.pyplot as plt

bdf_path = r"C:\Users\NTC\Documents\Bio-Fabulous-Unicorn\Recordings\Flicker recordings\Sin_Dry_Neutral\UnicornRecorder_25_02_2026_15_18_46.bdf"
raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False).pick("eeg")

raw.notch_filter([50], verbose=False)
raw.filter(l_freq=0.1, h_freq=None, verbose=False)
raw.set_eeg_reference("average", verbose=False)

# pick a stable segment (adjust)
raw_seg = raw.copy().crop(tmin=10.0, tmax=min(190.0, raw.times[-1]))

spectrum = raw_seg.compute_psd(method="welch", fmin=1, fmax=40,
                               n_fft=int(raw.info["sfreq"]*4),
                               n_overlap=int(raw.info["sfreq"]*2),
                               verbose=False)
psd, freqs = spectrum.get_data(return_freqs=True)   # (n_ch, n_freq)

def snr_spectrum(psd, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1, eps=1e-20):
    k = np.concatenate([np.ones(noise_n_neighbor_freqs),
                        np.zeros(2*noise_skip_neighbor_freqs+1),
                        np.ones(noise_n_neighbor_freqs)])
    k /= k.sum()
    mean_noise = np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"),
                                     axis=-1, arr=psd)
    edge = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    mean_noise = np.pad(mean_noise, [(0,0)]*(mean_noise.ndim-1)+[(edge,edge)],
                        constant_values=np.nan)
    return psd / (mean_noise + eps)

snr = snr_spectrum(psd)
psd_db = 10*np.log10(psd.mean(axis=0))
snr_mean = np.nanmean(snr, axis=0)

# Print SNR at your control freqs to see what "idle" looks like
targets = [8.5, 10.0, 12.0, 13.5]
print("Idle SNR near targets:")
for f in targets + [2*x for x in targets]:
    idx = np.argmin(np.abs(freqs - f))
    print(f"  {f:5.1f} Hz -> {freqs[idx]:6.2f} Hz : {snr_mean[idx]:6.2f}")

plt.figure(figsize=(9,4))
plt.plot(freqs, psd_db)
plt.xlim(1, 40)
plt.title("Baseline PSD (center fixation)")
plt.xlabel("Hz"); plt.ylabel("dB")
plt.show(block=False)

plt.figure(figsize=(9,4))
plt.plot(freqs, snr_mean)
plt.xlim(1, 40)
plt.title("Baseline SNR (center fixation)")
plt.xlabel("Hz"); plt.ylabel("SNR")
plt.show(block=True)