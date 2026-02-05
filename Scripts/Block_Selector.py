# вырезание сегментов по выбранному типу блоков из Experiment.json
# Обобщённая версия: block_type, t_plus, t_minus задаются параметрами.

import os
import json
import ast
import re
from functools import partial
from multiprocessing import Pool, cpu_count

import mne
import numpy as np

os.chdir('..')

# ============== Параметры (настраиваются пользователем) ==============
CPU_CORES = 30

BLOCK_TYPE = "pattern"   # тип блока из Experiment.json: "execution", "rest", "pattern", "command" и т.д.
T_PLUS = 0.5               # секунды после конца блока
T_MINUS = 0.5               # секунды до начала блока

ROOT_DIR = "./Generated/Data"
OUTPUT_DIR_SUFFIX = f"Data_Pattern"   # папка вывода: Generated/Data_Train(execution)/...
FILE_PREFIX = BLOCK_TYPE[:4] if len(BLOCK_TYPE) >= 4 else BLOCK_TYPE   # префикс файлов: exec_EEG_1.fif, rest_EEG_1.fif, ...

# Регулярки для S_ и Trial_ директорий
S_DIR_PATTERN = re.compile(r"^S_\d+$")
TRIAL_DIR_PATTERN = re.compile(r"^Trial_\d+$")



def get_blocks_times(experiment_seq, block_type, t_minus, t_plus):
    """
    Возвращает список времён сегментов для вырезания с учётом t_minus и t_plus.

    Для каждого блока типа block_type возвращается пара (crop_start_sec, crop_end_sec):
    - crop_start_sec = timestamp блока - t_minus
    - crop_end_sec   = timestamp блока + duration + t_plus

    Parameters
    ----------
    experiment_seq : dict
        Содержимое Experiment.json (ключ — id блока, значение — блок с 'type', 'timestamp', 'content']['duration']).
    block_type : str
        Тип блоков для выбора (например "execution", "rest").
    t_minus : float
        Секунды до начала блока, включаемые в сегмент.
    t_plus : float
        Секунды после конца блока, включаемые в сегмент.

    Returns
    -------
    list of tuple
        [(crop_start_sec, crop_end_sec), ...] — границы по времени в секундах для каждого сегмента.
    """
    out = []
    for block_data in experiment_seq.values():
        if block_data.get("type") != block_type:
            continue
        ts = block_data["timestamp"]
        dur = block_data["content"]["duration"]
        crop_start = ts - t_minus
        crop_end = ts + dur + t_plus
        out.append((crop_start, crop_end))
    return out


def collect_valid_paths(root_dir):
    """Собирает пути ко всем подходящим trial-директориям внутри root_dir (S_1/Trial_1, ...)."""
    valid_dirs = []
    for s_dir in os.listdir(root_dir):
        s_path = os.path.join(root_dir, s_dir)
        if not os.path.isdir(s_path) or not S_DIR_PATTERN.match(s_dir):
            continue
        for trial_dir in os.listdir(s_path):
            trial_path = os.path.join(s_path, trial_dir)
            if not os.path.isdir(trial_path) or not TRIAL_DIR_PATTERN.match(trial_dir):
                continue
            if any(os.path.isfile(os.path.join(trial_path, f)) for f in os.listdir(trial_path)):
                valid_dirs.append(trial_path)
    return valid_dirs


def _get_geometric_img(geometric_patterns, pattern_id):
    """Возвращает img по pattern_id. geometric_patterns — список (индекс = pattern_id) или словарь."""
    if pattern_id is None or not isinstance(pattern_id, (int, float)):
        return []
    pattern_id = int(pattern_id)
    if isinstance(geometric_patterns, list):
        if 0 <= pattern_id < len(geometric_patterns):
            return geometric_patterns[pattern_id]
        return []
    if isinstance(geometric_patterns, dict):
        return geometric_patterns.get(pattern_id, [])
    return []


def _build_labels_blocks(experiment_seq, block_type, geometric_patterns_path, gen_img):
    """Формирует список блоков для labels.json по experiment_seq и block_type."""
    blocks_for_type = [
        b for b in experiment_seq.values()
        if b.get("type") == block_type
    ]
    result_blocks = []
    geometric_patterns = []  # список паттернов по индексу (pattern_id)
    if geometric_patterns_path and os.path.isfile(geometric_patterns_path):
        with open(geometric_patterns_path, "r", encoding="utf-8") as f:
            geometric_patterns = ast.literal_eval(f.read())

    for i, block in enumerate(blocks_for_type, start=1):
        content = block.get("content", {})
        
        # execution: pattern_type и pattern_id заданы в content после препроцессинга
        if block_type == "execution" and content:
            pattern_type = content.get("pattern_type") # "geometric" или "random"
            pattern_id = content["pattern_id"] if "pattern_id" in content else content.get("seed")
            entry = {"block_index": i, "type": pattern_type}
            if pattern_type == "geometric":
                entry["pattern_id"] = pattern_id
                entry["img"] = _get_geometric_img(geometric_patterns, pattern_id)
            elif pattern_type == "random":
                entry["seed"] = pattern_id
                entry["img"] = gen_img(pattern_id) if (gen_img and pattern_id is not None) else []
            else:
                entry["block_index"] = i
                entry["type"] = block_type
                entry["content"] = content

        # pattern: type и pattern_id/seed лежат в content напрямую — формат как у execution
        elif block_type == "pattern" and content:
            pattern_type = content.get("type")  # "geometric" или "random"
            pattern_id = content["pattern_id"] if "pattern_id" in content else content.get("seed")
            entry = {"block_index": i, "type": pattern_type}
            if pattern_type == "geometric":
                entry["pattern_id"] = pattern_id
                entry["img"] = _get_geometric_img(geometric_patterns, pattern_id)
            elif pattern_type == "random":
                entry["seed"] = pattern_id
                entry["img"] = gen_img(pattern_id) if (gen_img and pattern_id is not None) else []
            else:
                entry["block_index"] = i
                entry["type"] = block_type
                entry["content"] = content
                
        else:
            entry = {"block_index": i, "type": block_type, "content": content}
            
        result_blocks.append(entry)
    return result_blocks


def preprocess(core_path, block_type, t_minus, t_plus, output_dir_suffix, file_prefix):
    s_name = os.path.basename(os.path.dirname(core_path))
    trial_name = os.path.basename(core_path)

    save_path_all = os.path.join(".", "Generated", "Data", s_name, trial_name)
    save_path_training = os.path.join(".", "Generated", output_dir_suffix, s_name, trial_name)

    eeg_clean_path = os.path.join(save_path_all, "EEG_clean.fif")
    eye_clean_path = os.path.join(save_path_all, "EOG_clean.fif")
    exp_seq_path = os.path.join(save_path_all, "Experiment.json")

    if not os.path.isfile(exp_seq_path):
        print(f"Skip (no Experiment.json): {core_path}")
        return
    if not os.path.isfile(eeg_clean_path) or not os.path.isfile(eye_clean_path):
        print(f"Skip (no EEG/EOG clean): {core_path}")
        return

    eeg_raw = mne.io.read_raw_fif(eeg_clean_path, preload=True, verbose="ERROR")
    sr = eeg_raw.info["sfreq"]
    eye_raw = mne.io.read_raw_fif(eye_clean_path, preload=True, verbose="ERROR")
    with open(exp_seq_path, "r", encoding="utf-8") as f:
        experiment_seq = json.load(f)

    block_times = get_blocks_times(experiment_seq, block_type, t_minus, t_plus)
    if not block_times:
        print(f"Skip (no blocks of type '{block_type}'): {core_path}")
        return

    os.makedirs(save_path_training, exist_ok=True)
    raw_duration_sec = len(eeg_raw) / sr - 0.001

    for i, (crop_start, crop_end) in enumerate(block_times):
        crop_start = max(0.0, crop_start)
        crop_end = min(crop_end, raw_duration_sec)
        if crop_end <= crop_start:
            continue

        eeg_segment = eeg_raw.copy().crop(tmin=crop_start, tmax=crop_end, verbose=False)
        eye_segment = eye_raw.copy().crop(tmin=crop_start, tmax=crop_end, verbose=False)

        eeg_segment.save(
            os.path.join(save_path_training, f"{file_prefix}_EEG_{i + 1}.fif"),
            overwrite=True,
            verbose="ERROR",
        )
        eye_segment.save(
            os.path.join(save_path_training, f"{file_prefix}_EOG_{i + 1}.fif"),
            overwrite=True,
            verbose="ERROR",
        )

    def gen_img(seed):
        seed = seed % (2 ** 32)
        np.random.seed(seed)
        img = 1 - np.random.randint(0, 2, size=(6, 6))
        return img.tolist()

    geometric_patterns_path = os.path.join(".", "Supplementary", "geometric_patterns.txt")
    result_blocks = _build_labels_blocks(
        experiment_seq, block_type, geometric_patterns_path, gen_img
    )
    result = {"blocks": result_blocks}
    labels_path = os.path.join(save_path_training, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as outfile:
        json.dump(result, outfile, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    
    paths = collect_valid_paths(ROOT_DIR)
    print(f"Block type: {BLOCK_TYPE}, t_minus={T_MINUS}s, t_plus={T_PLUS}s")
    print(f"Найдено {len(paths)} подходящих директорий. Запускаю обработку...")

    preprocess_fn = partial(
        preprocess,
        block_type=BLOCK_TYPE,
        t_minus=T_MINUS,
        t_plus=T_PLUS,
        output_dir_suffix=OUTPUT_DIR_SUFFIX,
        file_prefix=FILE_PREFIX,
    )
    with Pool(processes=CPU_CORES) as pool:
        pool.map(preprocess_fn, paths)

    print("Done!")
