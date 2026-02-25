import mne

bdf_path = r"C:\Users\NTC\Documents\Bio-Fabulous-Unicorn\Recordings\Flicker recordings\Sin_SemiWet_OSCAR+\Sin\UnicornRecorder_24_02_2026_17_08_35.bdf"

raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose=False)

# Find trigger events
events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose=False)

if len(events) == 0:
    print("No events found on Status channel.")
    raise SystemExit

sfreq = raw.info["sfreq"]
codes = events[:, 2].tolist()
times = (events[:, 0] / sfreq).tolist()

print("Trigger order (code @ time_s):")
for c, t in zip(codes, times):
    print(f"{c} @ {t:.3f}s")

# Optional: map codes to meaning
code_to_label = {
    1: "UP (13.5 Hz)",
    2: "RIGHT (8.5 Hz)",
    3: "DOWN (12 Hz)",
    4: "LEFT (10 Hz)",
}

print("\nTrigger order (labels):")
for c in codes:
    print(code_to_label.get(int(c), f"CODE {c}"))