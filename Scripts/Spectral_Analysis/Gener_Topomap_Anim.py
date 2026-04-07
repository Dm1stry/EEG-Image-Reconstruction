import mne
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import shutil
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import re

os.chdir('../..')

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")






def collect_valid_paths(root_dir):
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


    
def generate_topomap_gif(eeg, temp_dir_path, out_gif_path, 
                         freq, offset, 
                         duration, descretization, 
                         fps):
    """
    freq - Hz: target freq
    offset - Hz: region/window size (both sides) from target freq
    duration - S
    descretization - S
    fps - frames per sec
    """
    times = np.arange(0, duration, descretization)
    fmin, fmax = freq - offset, freq + offset
    
    # Создание папки для кадров
    os.makedirs(temp_dir_path, exist_ok=True)
    os.makedirs(os.path.dirname(out_gif_path), exist_ok=True)
    
    # Загрузка EEG
    raw = eeg
    
    # Сначала получим min и max значений по всем окнам, чтобы зафиксировать цветовую шкалу
    all_psds = []
    for t_start in times:
        t_end = t_start + 1
        raw_segment = raw.copy().crop(tmin=t_start, tmax=t_end)
        psds, freqs = raw_segment.compute_psd(fmin=fmin, fmax=fmax, verbose=False).get_data(return_freqs=True)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        psds_band = psds[:, freq_mask].mean(axis=1)
        all_psds.append(psds_band)
    
    all_psds = np.array(all_psds)
    vmin = np.min(all_psds)
    vmax = np.max(all_psds)
    
    # Генерация топомапов с colorbar и фиксированной шкалой
    frame_paths = []
    
    for i, psds_band in enumerate(all_psds):
        t_start = times[i]
        t_end = t_start + descretization
    
        fig, ax = plt.subplots(figsize=(5, 4))
        im, _ = mne.viz.plot_topomap(
            psds_band, raw.info, axes=ax, show=False,
            cmap='plasma', contours=0, vlim=(vmin, vmax)
        )
        ax.set_title(f'{t_start}-{t_end}s')
    
        # Добавим colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
        cbar.set_label('Power (a.u.)')
    
        frame_path = os.path.join(temp_dir_path, f'frame_{i:02d}.png')
        plt.title(f"{freq}Hz Frequency Localization ({t_start:.2f} sec.)")
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)
        frame_paths.append(frame_path)
    
    # Создание GIF
    images = [Image.open(p) for p in frame_paths]
    images[0].save(out_gif_path, save_all=True, append_images=images[1:], duration=1000/fps, loop=0)
    
    shutil.rmtree(temp_dir_path)



def process_rest(args):
    i, core_path = args 
    s_name = os.path.basename(os.path.dirname(core_path))
    trial_name = os.path.basename(core_path)

    exec_id = 9
    freq = 11
    offset = 2
    duration = 20
    descretization = 0.05
    fps = 20
    
    eeg_clean_path = f"./Generated/Data/{s_name}/{trial_name}/EEG_clean.fif"
    temp_dir = f"./Generated/Figures/Temp{i}_Gif_Frames"
    gif_path = f"./Generated/Figures/Spectral_Analysis/Rest/Animations/{s_name}_{trial_name}_Rest_Topomap_{freq}Hz.gif"

    # Load the .fif file
    raw = mne.io.read_raw_fif(eeg_clean_path, preload=True, verbose=False)
    raw.pick_types(eeg=True, verbose=False)  # только EEG-каналы
    
    rest_start = 370
    rest_end = 400
    segment = raw.copy().crop(tmin=rest_start, tmax=rest_end)

    generate_topomap_gif(
        eeg=segment,
        temp_dir_path=temp_dir,
        out_gif_path=gif_path,
        freq=freq,
        offset=offset,
        duration=duration,
        descretization=descretization,
        fps=fps,
    )









paths = collect_valid_paths("./Generated/Data")
print(f"Найдено {len(paths)} подходящих директорий. Запускаю обработку...")

# число воркеров = числу ядер
with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    list(tqdm(
        executor.map(process_rest, enumerate(paths)),
        total=len(paths),
        desc="Rendering Topomap gifs",
    ))