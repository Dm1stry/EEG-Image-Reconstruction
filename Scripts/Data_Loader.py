import json
import numpy as np
import pandas as pd
import mne
import re
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import torch
from torch.utils.data import Dataset as TorchDataset
from glob import glob



class EIRDataset:
    def __init__(self, dataset_path, task_type='all', n_jobs=8):
        
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.n_jobs = n_jobs
        
        self.exp_paths = self.__collect_valid_paths(dataset_path)
        self.suply_data = pd.DataFrame(columns=["index", "subject_id", "trial_id", "task_type"])
        self.eeg_arr = []
        self.eye_arr = []
        self.labels = []
        self.imgs = []

        self.__load_dataset()


    def __collect_valid_paths(self, root_dir):
        valid_dirs = []
        
        # Регулярка для S_директорий: S_ и только цифры после
        s_dir_pattern = re.compile(r"^S_\d+$")
        # Регулярка для Trial_директорий: Trial_ и только цифры
        trial_dir_pattern = re.compile(r"^Trial_\d+$")
        
        for s_dir in os.listdir(root_dir):
            s_path = os.path.join(root_dir, s_dir)
            if os.path.isdir(s_path) and s_dir_pattern.match(s_dir):
                for trial_dir in os.listdir(s_path):
                    trial_path = os.path.join(s_path, trial_dir)
                    if os.path.isdir(trial_path) and trial_dir_pattern.match(trial_dir):
                        # Проверяем, есть ли хотя бы один файл
                        if any(os.path.isfile(os.path.join(trial_path, f)) for f in os.listdir(trial_path)):
                            valid_dirs.append(trial_path)
        return valid_dirs

    
    def load_fif_wrapper(self, args):
        eeg_path, eye_path, index = args

        eeg = mne.io.read_raw_fif(eeg_path, preload=True, verbose='ERROR')
        eye = mne.io.read_raw_fif(eye_path, preload=True, verbose='ERROR')

        return eeg, eye


    def __load_dataset(self):
        all_tasks = []
        index = 0
    
        for exp_path in tqdm(self.exp_paths, desc="Processing exp_paths"):
            subject_id = int(os.path.basename(os.path.dirname(exp_path)).replace("S_", ""))
            trial_id = int(os.path.basename(exp_path).replace("Trial_", ""))
    
            with open(os.path.join(exp_path, "labels.json"), "r") as f:
                labels_data = json.load(f)["blocks"]
    
            for block in labels_data:
                task_type = block["type"]
                if task_type == self.task_type or self.task_type == 'all':
                    exec_idx = block["Exec_Block_Index"]
                    pattern_id = block.get("pattern_id", -1)
    
                    # search files by the template <any_text>_EEG_<index>.fif and <any_text>_EOG_<index>.fif 
                    eeg_candidates = glob(os.path.join(exp_path, f"*_EEG_{exec_idx}.fif"))
                    eog_candidates = glob(os.path.join(exp_path, f"*_EOG_{exec_idx}.fif"))
    
                    if not eeg_candidates or not eog_candidates:
                        print(f"Skipping: EEG or EOG file missing for exec_idx={exec_idx} in {exp_path}")
                        continue
    
                    # Get first candidate (mostly there is only 1 candidate)
                    eeg_path = eeg_candidates[0]
                    eye_path = eog_candidates[0]
    
                    self.suply_data.loc[len(self.suply_data)] = {
                        "index": index,
                        "subject_id": subject_id,
                        "trial_id": trial_id,
                        "task_type": task_type
                    }
    
                    self.labels.append(pattern_id)
                    self.imgs.append(np.array(block["img"]))
                    all_tasks.append((eeg_path, eye_path, index))
                    index += 1


        # Параллельная загрузка данных
        with Pool(self.n_jobs) as pool:
            results = list(tqdm(pool.imap(self.load_fif_wrapper, all_tasks), total=len(all_tasks), desc="Loading .fif files"))

        # Распакуем результаты
        for eeg, eye in results:
            self.eeg_arr.append(eeg)
            self.eye_arr.append(eye)

    
    def __getitem__(self, idx):
        return (
            self.eeg_arr[idx],       # mne.Raw EEG
            self.eye_arr[idx],       # mne.Raw EYE
            self.suply_data.iloc[idx].to_dict(),
            self.labels[idx],        # pattern_id
            self.imgs[idx]           # numpy 6x6 array
        )

    def __setitem__(self, idx, value):
        eeg_sample, eye_sample, metadata_dict, label, img = value

        self.eeg_arr[idx] = eeg_sample
        self.eye_arr[idx] = eye_sample
        self.suply_data.iloc[idx] = metadata_dict  # если нужно — или опустить, если не редактируется
        self.labels[idx] = label
        self.imgs[idx] = img

        
    def __len__(self):
        return len(self.suply_data)

    
    def to_torch(self):
        class TorchEIRDataset(TorchDataset):
            def __init__(self, eeg_arr, eye_arr, metadata, labels, imgs):
                self.eeg_arr = [e.get_data() for e in eeg_arr]
                self.eye_arr = [e.get_data() for e in eye_arr]
                self.metadata = metadata
                self.labels = labels
                self.imgs = imgs

            def __getitem__(self, idx):
                return (
                    torch.tensor(self.eeg_arr[idx], dtype=torch.float32),
                    torch.tensor(self.eye_arr[idx], dtype=torch.float32),
                    self.metadata.iloc[idx].to_dict(),
                    torch.tensor(self.labels[idx], dtype=torch.long),
                    torch.tensor(self.imgs[idx], dtype=torch.float32)
                )

            def __len__(self):
                return len(self.labels)

        return TorchEIRDataset(self.eeg_arr, self.eye_arr, self.suply_data, self.labels, self.imgs)

        
    # def to_tensorflow(self):
    #     eeg_data = np.array([e.get_data() for e in self.eeg_arr], dtype=np.float32)
    #     eye_data = np.array([e.get_data() for e in self.eye_arr], dtype=np.float32)
    #     labels = np.array(self.labels, dtype=np.int64)
    #     imgs = np.array(self.imgs, dtype=np.float32)

    #     def gen():
    #         for i in range(len(labels)):
    #             yield eeg_data[i], eye_data[i], labels[i], imgs[i]

    #     return tf.data.Dataset.from_generator(
    #         gen,
    #         output_signature=(
    #             tf.TensorSpec(shape=eeg_data.shape[1:], dtype=tf.float32),
    #             tf.TensorSpec(shape=eye_data.shape[1:], dtype=tf.float32),
    #             tf.TensorSpec(shape=(), dtype=tf.int64),
    #             tf.TensorSpec(shape=(6, 6), dtype=tf.float32)
    #         )
    #     )

    
    # def to_catboost(self):
    #     # Плоские признаки (средние по времени)
    #     features = []
    #     for eeg, eye in zip(self.eeg_arr, self.eye_arr):
    #         eeg_feat = np.mean(eeg.get_data(), axis=1)
    #         eye_feat = np.mean(eye.get_data(), axis=1)
    #         features.append(np.concatenate([eeg_feat, eye_feat]))
    #     features = np.array(features)

    #     labels = np.array(self.labels)
    #     return CatBoostPool(data=features, label=labels)
