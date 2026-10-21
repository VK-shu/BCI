"""Generate FAKE BrainLink data, only to test that main.py runs when no real data is available.
The spectral shapes are hand-made, so accuracy on it says nothing about real performance.
Do NOT include it in the submitted zip.

Usage:  python gen_fake_data.py [output folder, default fake_dataset]
"""
import os, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt

FS, N = 512, 10240
rng = np.random.default_rng(0)

def band_noise(lo, hi, n=N):
    sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal(n + 2000))[1000:-1000]
    return x / x.std()

def make_block(task, gain, dc):
    base = 1.0 * band_noise(1, 45) + 0.6 * band_noise(0.5, 4)   # background
    if task == 1:   # Relax: strong alpha, weak beta/gamma
        x = base * 0.7 + 1.6 * band_noise(8, 13) + 0.4 * band_noise(20, 40)
    elif task == 2: # Focus: strong beta/gamma, weak delta/theta/alpha
        x = base * 0.5 + 1.6 * band_noise(20, 40) + 0.2 * band_noise(8, 13)
    else:           # Stroop: strong theta/delta, moderate beta
        x = base * 0.7 + 1.5 * band_noise(4, 8) + 0.8 * band_noise(20, 40)
    x = x * 40 * gain + dc + rng.normal(0, 3, N)
    # occasionally add a blink-like large wave
    if rng.random() < 0.15:
        p = rng.integers(0, N - 400)
        x[p:p + 400] += 250 * np.hanning(400) * rng.choice([-1, 1])
    return np.round(x).astype(int)

def main(out):
    subjects = {"b00000001": (1.0, 50), "b00000002": (1.3, 60), "b00000003": (0.8, 40)}
    for sid, (gain, dc) in subjects.items():
        d = os.path.join(out, sid)
        os.makedirs(d, exist_ok=True)
        for task in (1, 2, 3):
            for rnd in range(1, 11):
                g = gain * rng.uniform(0.9, 1.1)
                np.savetxt(os.path.join(d, f"{sid}_{task}_{rnd}.txt"),
                           make_block(task, g, dc), fmt="%d")
    print("done ->", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fake_dataset")
