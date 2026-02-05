# Spectrum Files Description

This directory contains precomputed spectral representations of EEG data, stored as NumPy compressed archives (`.npz`). The files are used for spectral analysis, clustering, classification, and reconstruction in the EEG-Visual-Experiment pipeline. You can download all the files from ...TBA...

---

## Summary Table

| File | N Samples | Shape | Freq. range | Freq. resolution | Sample length | Time resolution |
|------|-----------|-------|-------------|------------------|----------------|-----------------|
| `exec_and_rest_fft.npz` | ...TODO... | (63, 381) | 2–40 Hz | 0.1 Hz | 20.5 s | — |
| `exec_and_rest_morlets.npz` | ...TODO... | (63, 195, 411) or (63, 195, 310) | 2–40 Hz | 0.2 Hz | 20.5 s | 50 ms |
| `exec_fft.npz` | ...TODO... | (63, 381) | 2–40 Hz | 0.1 Hz | 16 s | — |
| `exec_morlets.npz` | ...TODO... | (63, 195, 321) or (63, 195, 310) | 2–40 Hz | 0.2 Hz | 16 s | 50 ms |
| `psds_array_fft.npz` | ...TODO... | (63, 381) | 2–40 Hz | 0.1 Hz | Full experiment | — |
| `psds_array_morlet.npz` | ...TODO... | (63, 80) | 2–40 Hz | 0.5 Hz | Full experiment | — |
| `pattern_fft.npz` | ...TODO... | (63, 381) | 2–40 Hz | 0.1 Hz | 26 s | — |
| `pattern_morlets.npz` | ...TODO... | (63, 195, 520) | 2–40 Hz | 0.2 Hz | 26 s | 50 ms |
---

**Note:** For FFT files, time resolution is absent due to the limitations of the method.

---

### 1. `exec_and_rest_fft.npz`

**Fourier spectra: target execution blocks + short rest interval after each block.**

- **Spectrum shape:** `(63, 381)`  
  - 63: number of electrodes  
  - 381: number of frequency bins  
- **Sample duration:** 20.5 s (target block + following rest)

Suitable for frequency-only analysis when both execution and the immediate post-execution rest are included in each segment.

---

### 2. `exec_and_rest_morlets.npz`

**Morlet wavelet spectra: target execution blocks + short rest interval after each block.**

- **Spectrum shape:** `(63, 195, 411)` or `(63, 195, 310)`  
  - 63: number of electrodes  
  - 195: number of frequency bins  
  - 411 or 310: number of time points  
- **Sample duration:** 20.5 s

Provides time–frequency representations for the same segments as `exec_and_rest_fft.npz`, allowing analysis of how power evolves during execution and early rest. The third dimension may vary (411 vs 310) depending on "last block in experiment" condition.

---

### 3. `exec_fft.npz`

**Fourier spectra: target execution blocks only (no rest).**

- **Spectrum shape:** `(63, 381)`  
  - 63: number of electrodes  
  - 381: number of frequency bins  
- **Sample duration:** 16 s

Same frequency setup as the FFT exec+rest file, but segments contain only the 16 s execution interval. Use when rest should be excluded from the spectrum.

---

### 4. `exec_morlets.npz`

**Morlet wavelet spectra: target execution blocks only (no rest).**

- **Spectrum shape:** `(63, 195, 321)` or `(63, 195, 310)`  
  - 63: number of electrodes  
  - 195: number of frequency bins  
  - 321 or 310: number of time points  
- **Sample duration:** 16 s

Time–frequency representation restricted to the 16 s execution window. The third dimension may vary (321 vs 310) depending on "last block in experiment" condition.

---

### 5. `psds_array_fft.npz`

**FFT spectras over the full experiment.**

- **Spectrum shape:** `(63, 381)`  
  - 63: number of electrodes  
  - 381: number of frequency bins  
- **Sample duration:** Full experiment (exact duration to be documented)

Each entry is a PSD summarizing one complete record over the whole experiment. Useful for global spectral characterisation and between-subject or between-condition comparisons.

---

### 6. `psds_array_morlet.npz`

**Morlet wavelets spectras over the full experiment.**

- **Spectrum shape:** `(63, 80)`  
  - 63: number of electrodes  
  - 80: number of frequency bins  
- **Sample duration:** Full experiment (exact duration to be documented)

Same role as `psds_array_fft.npz` but with Morlet-based PSDs and coarser frequency resolution (0.5 Hz, 80 bins in 2–40 Hz). Use when preferring wavelet-derived PSDs or when lower frequency resolution is sufficient.

---

### 7. `pattern_fft.npz`

**Fourier spectra: pattern blocks only.**

- **Spectrum shape:** `(63, 381)`  
  - 63: number of electrodes  
  - 381: number of frequency bins  
- **Sample duration:** 26 s  
- **Frequency resolution:** 0.1 Hz  

FFT spectra for fixed-length 26s pattern block (when subjec see the pattern) +-0.5s. Same frequency setup as other FFT files (2–40 Hz, 0.1 Hz). Use for frequency-only analysis of pattern intervals.

---

### 8. `pattern_morlets.npz`

**Morlet wavelet spectra: pattern blocks only.**

- **Spectrum shape:** `(63, 195, 520)` (third dimension may vary)  
  - 63: number of electrodes  
  - 195: number of frequency bins  
  - 520: number of time points (26 s ÷ 50 ms)  
- **Sample duration:** 26 s  
- **Time resolution:** 50 ms  
- **Frequency resolution:** 0.2 Hz  

Time–frequency representation for the 26s pattern block (when subjec see the pattern) +-0.5s as in `pattern_fft.npz`, with 50 ms time steps and 0.2 Hz frequency steps. Use when temporal evolution of power within the pattern window is needed.

---

## Usage Notes

You can load any of these files in Python using the following code:

```python
import numpy as np

loaded = np.load('./path/to/array.npz')

results_arr = []

i = 0
while f'power_{i}' in loaded:
    psd = loaded[f'power_{i}']
    s_id = int(loaded[f'subject_id_{i}'])
    t_id = int(loaded[f'trial_id_{i}'])
    gender = str(loaded[f'gender_{i}'])
    handiness = str(loaded[f'handiness_{i}'])
    age = int(loaded[f'age_{i}'])
    
    results_arr.append([psd, s_id, t_id, gender, handiness, age])
    i += 1

psd, s_id, t_id, gender, handiness, age = results_arr[0]
psd.shape
```

- **FFT vs Morlet:** FFT files give one spectrum per segment (no time axis). Morlet files add a time dimension for time–frequency analysis.
- **Execution vs exec+rest:** Use `exec_*` when only the execution interval matters; use `exec_and_rest_*` when the short rest after the block should be included.
- **Full-experiment PSDs:** `psds_array_fft.npz` and `psds_array_morlet.npz` are for experiment-level or subject-level summary spectra; segment duration corresponds to the full experiment (TODO: document exact duration).
