from torch.utils.data import Dataset as TorchDataset
import torch
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, List
import numpy as np
import json
from tqdm import tqdm
import mne
from multiprocessing import Pool, cpu_count
from enum import Enum
import os

class LazyLoadMorletDataset(TorchDataset):
    dataset_path: Path
    morlet_data_path: Path
    idx_to_record: List[Tuple[int, int, int]] #Maps index in dataset to (Subject_id, Trial_id, Sample_id)

    subject_dir_pattern = 'S_*'
    trial_dir_pattern = 'Trial_*'
    sample_eeg_pattern = 'exec_EEG_*'
    sample_eog_pattern = 'exec_EOG_*'

    def __init__(self, 
                 dataset_path: Path | str, 
                 morlet_data_path: Path | str, 
                 task_type: str = 'all', 
                 eeg_resampling_freq: int | None = None, 
                 max_morlet_wav_len: int | None = 309, 
                 samples_to_exclude: Tuple[int] = (), 
                 subjects_to_exclude: Tuple[int] = (), 
                 n_jobs: int = 8, 
                 eog_needed: bool = True,
                 wavelet_needed: bool = True,
                 max_eeg_raw_length: int | None = None, 
                 max_eog_raw_length: int | None = None):
        
        self.dataset_path = Path(dataset_path)
        self.morlet_data_path = Path(morlet_data_path)
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.eog_needed = eog_needed
        self.wavelet_needed = wavelet_needed
        self.max_eeg_raw_length = max_eeg_raw_length
        self.max_eog_raw_length = max_eog_raw_length
        self.idx_to_record = []

        self.eeg_resampling_freq = eeg_resampling_freq
        self.max_morlet_wav_len = max_morlet_wav_len
        
        self.samples_to_excluse = samples_to_exclude
        self.subjects_to_exclude = subjects_to_exclude
        
        self.idx_to_record = self.__collect_valid_trial_paths(self.dataset_path)
        self.labels, self.images, self.metadata = self.__load_all_metadata(self.dataset_path, self.idx_to_record)


    def __getitem__(self, index):
        subject_index, trial_index, sample_index = self.idx_to_record[index]
        trial_path = self.get_trial_subpath_by_ids(subject_index, trial_index)
        eeg_sample_path = self.dataset_path / trial_path / (self.sample_eeg_pattern.replace("*", str(sample_index), 1) + ".fif")
        eog_sample_path = self.dataset_path / trial_path / (self.sample_eog_pattern.replace("*", str(sample_index), 1) + ".fif")
        morlet_sample_path = self.morlet_data_path / trial_path / f"exec_morlets_{subject_index}_{trial_index}_{sample_index}.npz"

        EEG = self.__load_eeg(eeg_sample_path, self.eeg_resampling_freq, self.max_eeg_raw_length)
        
        if self.eog_needed:
            EOG = self.__load_eog(eog_sample_path, self.max_eog_raw_length)

        if self.wavelet_needed:
            power_wav, phase_wav = self.__load_morlets(morlet_sample_path)

        label = self.labels[index]
        image = self.images[index]
        metadata = self.metadata.iloc[index]

        if self.eog_needed and self.wavelet_needed:
            return EEG, EOG, power_wav, phase_wav, label, image, metadata
        elif self.eog_needed:
            return EEG, EOG, label, image, metadata
        elif self.wavelet_needed:
            return EEG, power_wav, phase_wav, label, image, metadata
        else:
            return EEG, label, image, metadata

    def __len__(self):
        return len(self.idx_to_record)

    def __collect_valid_trial_paths(self, root_dir: str, check_eog_presense: bool = True):
        root_path = Path(root_dir)
        idx_to_record = []

        for s_path in root_path.glob(self.subject_dir_pattern):
            if not s_path.is_dir():
                continue
            current_subject_index = self.__get_index_from_dir_name(s_path, self.subject_dir_pattern)

            if current_subject_index in self.subjects_to_exclude:
                continue

            for trial_path in s_path.glob(self.trial_dir_pattern):
                if not trial_path.is_dir():
                    continue
                current_trial_index = self.__get_index_from_dir_name(trial_path, self.trial_dir_pattern)
                for eeg_sample_path in trial_path.glob(self.sample_eeg_pattern):
                    if not eeg_sample_path.is_file():
                        continue
                    current_sample_index = self.__get_index_from_dir_name(eeg_sample_path, self.sample_eeg_pattern)
                    eog_path = trial_path / self.sample_eog_pattern.replace("*", str(current_sample_index) + ".fif")
                    if check_eog_presense and not eog_path.exists():
                        continue
                    if current_sample_index in self.samples_to_excluse:
                        continue

                    idx_to_record.append((current_subject_index, current_trial_index, current_sample_index))
        idx_to_record.sort()
        return idx_to_record

    def get_index_by_ids(self, subject_id: int, trial_id: int, sample_id: int) -> int:
        triplet = (subject_id, trial_id, sample_id)
        if triplet not in self.idx_to_record:
            return None
        
        return self.idx_to_record.index(triplet)

    def get_trial_subpath_by_ids(self, subject_index: int, trial_index: int):
        return Path(self.subject_dir_pattern.replace("*", str(subject_index), 1)) / self.trial_dir_pattern.replace("*", str(trial_index), 1)
    
    def __get_index_from_dir_name(self, dir_path: Path, dir_pattern: str) -> int:
        return int(str(dir_path.name).replace(dir_pattern[:-1], "").replace(".fif", ""))
        
    def __load_eeg(self, eeg_path: Path | str, eeg_resampling_freq: int | None = None, max_length: int | None = None):
        eeg = mne.io.read_raw_fif(eeg_path, preload=True, verbose='ERROR')
        if max_length:
            eeg.crop(tmax=max_length / eeg.info['sfreq'], include_tmax=False)
        if eeg_resampling_freq is not None:
            eeg.resample(eeg_resampling_freq, n_jobs=self.n_jobs, verbose='ERROR')
        eeg = torch.from_numpy(eeg.get_data(verbose='ERROR'))

        return eeg

    def __load_eog(self, eog_path: Path | str, max_length: int | None = None):
        eog = mne.io.read_raw_fif(eog_path, preload=True, verbose='ERROR')
        eog = torch.from_numpy(eog.get_data(verbose='ERROR')[:, :max_length])

        return eog

    def __load_morlets(self, morlet_path: Path) -> Tuple[np.array, np.array]:
        loaded_morlet_data = dict(np.load(morlet_path))

        power_wav = np.array(loaded_morlet_data[f'power'][:, :, :self.max_morlet_wav_len])
        phase_wav = np.array(loaded_morlet_data[f'phase'][:, :, :self.max_morlet_wav_len])

        power_wav_combined = power_wav.transpose(1, 0, 2).reshape(power_wav.shape[0], power_wav.shape[1]*power_wav.shape[2])
        phase_wav_combined = phase_wav.transpose(1, 0, 2).reshape(phase_wav.shape[0], phase_wav.shape[1]*phase_wav.shape[2])

        return torch.from_numpy(power_wav_combined), torch.from_numpy(phase_wav_combined)

    # def __load_metadata(self, trial_path: Path, subject_id: int, trial_id: int):
    #     labels = None
    #     images = None
    #     metadata_collection = None

    #     with open(os.path.join(trial_path, "labels.json"), "r") as f:
    #         labels_data = json.load(f)["blocks"]
    #         for block in labels_data:
    #             task_type = block["type"]
    #             if self.task_type == task_type or self.task_type == 'all':
    #                 exec_idx = block["Exec_Block_Index"]

    #                 if exec_idx in self.samples_to_excluse:
    #                     continue

    #                 labels.append(block.get("pattern_id", -1))
    #                 images.append(np.array(block["img"], dtype=np.int8))

    #                 metadata_collection.append({
    #                     "subject_id": subject_id,
    #                     "trial_id": trial_id,
    #                     "task_type": task_type
    #                 })            

    #     return labels, images, metadata_collection
    

    def __load_all_metadata(self, dataset_path: Path, idx_to_record: List[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        labels = []
        images = []
        metadata = pd.DataFrame(columns=["subject_id", "trial_id", "task_type"])

        last_pair = None

        for subject_id, trial_id, sample_id in idx_to_record:
            if last_pair == (subject_id, trial_id):
                continue
            last_pair = (subject_id, trial_id)

            trial_path = dataset_path / self.get_trial_subpath_by_ids(subject_id, trial_id)

            with open(os.path.join(trial_path, "labels.json"), "r") as f:
                labels_data = json.load(f)["blocks"]

                for block in labels_data:
                    task_type = block["type"]
                    if task_type == self.task_type or self.task_type == 'all':
                        exec_idx = block["Exec_Block_Index"]

                        if exec_idx in self.samples_to_excluse:
                            continue

                        pattern_id = block.get("pattern_id", -1)

                        metadata.loc[len(metadata)] = {
                            "subject_id": subject_id,
                            "trial_id": trial_id,
                            "task_type": task_type
                        }
                        labels.append(pattern_id)
                        images.append(np.array(block["img"]))
            
        labels = torch.tensor(labels, dtype=torch.int8)
        images = torch.tensor(images, dtype=torch.int8)

        return labels, images, metadata
