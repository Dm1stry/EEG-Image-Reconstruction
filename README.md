# EEG-Visual-Experiment

A research project on **reconstructing imagined visual stimuli from EEG signals**. The goal is to learn to “read” and reconstruct images (geometric patterns) that a person imagines, from brain activity recordings. The experiment yields one of the largest EEG datasets on this topic.

Everything is described in detail in the corresponding scientific publication: ...TBA...

---

## Repository structure

**Only the following top-level directories are part of the repository:**

| Directory | Purpose | Contents |
|-----------|---------|----------|
| **Datasets** | Raw experiment data | Placeholder for the dataset; raw recordings go here (see note below). |
| **Generated** | Pipeline outputs | Preprocessed data, spectra, classification and reconstruction results, figures, etc. |
| **Notebooks** | Interactive analysis | `.ipynb` notebooks: loading data, preprocessing, spectral analysis, modeling, etc. |
| **Scripts** | Reusable code | `.py` modules and small scripts for loading, preprocessing, spectra, statistics, etc. |
| **Supplementary** | Auxiliary data | Experiment metadata, statistical tables, pattern definitions, etc. |

> **Note.** For **Datasets** and **Generated**, some files must be downloaded separately from ...TBA...

---

## Directory tree (overview)

```
EEG-Visual-Experiment/
├── Datasets/
│
├── Generated/
│   ├── Data/                          # Preprocessed EEG/EOG (gitignored)
│   ├── Data_Train/                    # Training windows: exec only (−0.5…+0.5 s)
│   ├── Data_Train(Exec_and_Rest)/     # Training windows: exec + rest (−0.5…+5.0 s)
│   ├── Spectrums/                     # NPZ spectra (FFT, Morlet wavelets)
│   ├── Figures/                       # Plots and reports (PNG, HTML)
│   └── Results/                       # Numeric results (JSON, CSV)
│
├── Notebooks/
│   ├── TODO                           # ... notebooks will be enumerated at the end
│   ├── Modeling/                      # Reconstruction/classification models
│   │   └── TODO                       # notebooks will be enumerated at the end
│   └── Spectral_Analysis/             # All spectrum-related analytics
│       ├── TODO                       # ... notebooks will be enumerated at the end
│       └── Statistical_Differences/   # Group-wise statistical comparisons
│           └── TODO                   # ... notebooks will be enumerated at the end
│
├── Scripts/
│   ├── Data_Loader.py                 # Module: load FIF, labels, patterns
│   ├── Statistical_Tester.py          # Module: band-power statistical testing
│   ├── Preprocessing.py               # Script: full-dataset preprocessing (like Preprocessing.ipynb)
│   ├── Rebuild_Train_Dataset.py       # Script: slice exec/rest training windows
│   ├── Selectors_From_Dataset.py      # Script: sample by subject/trial
│   ├── EEGModels.py                   # Module: EEG CNNs (EEGNet etc., Keras/TensorFlow)
│   └── Spectral_Analysis/             # Spectral analysis automation
│       └── TODO                       # ... will be described in the end
│
└── Supplementary/
    ├── Experiment_Metadata.xlsx       # Experiment metadata
    ├── geometric_patterns.txt         # Geometric stimuli (6×6 grids)
    └── band_table.csv                 # Frequency bands for exec blocks
```

---

## About the project

- **Task:** From multichannel EEG during imagination of geometric patterns — classify or reconstruct the image (a black/white 6×6 pixel grid).

- **Data:**  
  The dataset is organized by **subjects** and **trials**. Each **subject** has an id and lives in a folder `S_<id>` (e.g. `S_1`, `S_2`). Inside each subject folder there are **trials** in folders `Trial_<id>` (e.g. `Trial_1`, `Trial_2`). Each trial folder contains raw EEG, EOG, markers, and block descriptions.

  **Layout of `Datasets/Data`:**

  ```
  Datasets/Data/
  └── S_<id>/
      └── Trial_<id>/
            ├── EEG_Properties.json   # Sampling rate, channel names, per-channel resolution
            ├── EEG.csv               # Raw EEG samples (block id + channel columns)
            ├── Experiment.json       # Trial timeline: rest/exec blocks, timestamps, durations
            ├── Eyetracker.asc        # Eye-tracking / EOG (Eyelink format)
            └── EEG_Markers.csv       # Event markers (block id, position, channel, type)
  ```

  After preprocessing, data are saved as MNE FIF in `Generated/Data` and `Generated/Data_Train(Exec_and_Rest)` (also available for download ...TBA...). To work with the data yourself, use **`Scripts/Data_Loader.py`**; a detailed example is in **`Notebooks/Data_Load_Example.ipynb`**.

- **EEG:** 64 channels, 1000 Hz, standard channel names (Fp1, Fz, F3, …).

---

## Loading and preprocessing

TODO

---

## Spectral analysis

TODO

---

## Classification

TODO

---

## Reconstruction and advanced modeling

TODO

---

## Statistics

TODO

---

## Dependencies

Python 3.9.x

Install: `pip install -r requirements.txt`

---

## License

MIT License (see `LICENSE`).
