import os
import numpy as np
import pandas as pd
import re


def _get_day_time_df(day_time_meta_path):
    # Читаем первый лист
    meta1 = pd.read_excel(day_time_meta_path, sheet_name=0, header=1).rename(columns={
        'Subject ID'          : 'Subject_id',
        'Время начала записи' : 'Time',
    })
    meta1['Trial_id'] = 1
    
    # Читаем второй лист
    meta2 = pd.read_excel(day_time_meta_path, sheet_name=1, header=1).rename(columns={
        'Subject ID'          : 'Subject_id',
        'Время начала записи' : 'Time',
    })
    meta2['Trial_id'] = 2
    
    # Конкатенация
    meta = pd.concat([meta1, meta2], ignore_index=True)
    
    meta
    
    meta['Subject_id'] = (
        meta['Subject_id']
        .astype(str)
        .str.extract(r'(\d+)', expand=False)
        .astype('float')
    )
    meta = meta[meta['Subject_id'].notna()].copy()
    meta['Subject_id'] = meta['Subject_id'].astype(int)

    # Вот отсюда...
    s  = meta['Time'].astype(str).str.strip()                    
    n  = pd.to_numeric(s, errors='coerce')                        
    dt_str = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
    dt_num = pd.to_datetime(n, errors='coerce', origin='1899-12-30', unit='D')
    dt = dt_str.fillna(dt_num)                                   
    # ...и до сюда - GPT-кадавр который мало того что не должен существовать, так ещё и не работает нормально. Итог - все даты где день >= 12 не подтягиваются
    # UPD: нашёл почему это произошло, первый раз писалось для старых данных. А они писались с 1 по 10 числа, то есть на этапе первичной отладки это было не поймать
    
    meta['Hour'] = dt.dt.hour
    meta['Time'] = dt.dt.strftime('%H:%M')
    
    meta_idx = (meta[['Subject_id', 'Trial_id', 'Hour', 'Time']]
                .dropna(subset=['Hour'])                         
                .drop_duplicates(['Subject_id', 'Trial_id'], keep='last'))
    
    meta_idx['Condition'] = pd.cut(
        meta_idx['Hour'],
        bins=[-0.1, 10, 18, 24],
        labels=['Other', 'Day', 'Evening'],
        right=False,
        include_lowest=True
    ).astype(object).fillna('Other')

    return meta_idx



# Словарь возможных синонимов для каждого поля
FIELD_SYNONYMS = {
    'power': ['power', 'pwr', 'psd'],
    'phase': ['phase', 'phs', 'phse'],
    'subject_id': ['subject_id', 'sid', 's_id', 'subj_id'],
    'trial_id': ['trial_id', 'tid', 't_id'],
    'gender': ['gender', 'gend', 'gen'],
    'handiness': ['handiness', 'hand'],
    'age': ['age', 'ag'],
    'label': ['label', 'true_label', 'labl', 'lbl'],
    'img': ['img', 'image', 'picture', 'pattern', 'patrn', 'pictr'],
    'task_type': ['task_type', 'task', 'pattern_type', 'p_type', 'image_type']
}



def _normalize_name(name: str) -> str:
    """Приводим имя ключа к унифицированному виду: без символов, в нижнем регистре."""
    return re.sub(r'[^a-z0-9]', '', name.lower())



def _get_field(loaded: dict, base_field: str, i: int):
    """Находит поле по синонимам, номеру и без учёта регистра/символов."""
    norm_loaded = {_normalize_name(k): k for k in loaded.keys()}
    candidates = []
    for synonym in FIELD_SYNONYMS.get(base_field, []):
        # Возможные варианты имени (с индексом и без)
        candidates += [
            f"{synonym}_{i}",
            f"{synonym}{i}",
            f"{synonym}-{i}",
            f"{synonym}{ {i} }",  # иногда фигурные скобки
            synonym
        ]

    for cand in candidates:
        norm_cand = _normalize_name(cand)
        if norm_cand in norm_loaded:
            return loaded[norm_loaded[norm_cand]]

    return None  # если не найдено

    
def _safe_get_field(loaded, field, i, cast=None):
    """
    Безопасно достаёт поле по синонимам.
    Возвращает None, если не найдено или ошибка преобразования.
    """
    try:
        val = _get_field(loaded, field, i)
        if val is None:
            return None
        return cast(val) if cast else val
    except Exception:
        return None

        
def filter_spectras(exec_spec_path,
                     day_time=None,
                     day_time_meta_path=None,
                     stim_type=None, 
                     stim_label=None, 
                     gender=None, 
                     age=None, 
                     handiness=None):
    """
    Filters experimental PSD data (.npz file) based on subject and trial metadata.

    :param: exec_spec_path : str - Path to the `.npz` file containing experimental PSD data.
    :param: day_time : str - Daytime condition "Day" or "Evening"
    :param: day_time_meta_path : str - Path to an Excel file with metadata. Required only if `day_time` is provided.
    :param: stim_type : str or list(str) - "g" → geometric or "r" → random.
    :param: stim_label : int or list(int) - Label(s) of stimuli ("-1" → random or "0–12" → geometric).
    :param: gender : str - Gender "m" → male or  "f" → female.
    :param: age : int or list(int) - Age(s).
    :param: handiness : str - Hand preference "r" → right-handed or "l" → left-handed

    :return: list - filtered execution spectras [power, phase, subject_id, trial_id, gender, handiness, age, label, img, task_type].
    """
    
    loaded = np.load(exec_spec_path)
    results_arr = []

    # Load daytime metadata if filtering by daytime
    if day_time is not None:
        if day_time_meta_path is None:
            raise ValueError("You must provide 'day_time_meta_path' when 'day_time' is specified.")
        day_time_df = _get_day_time_df(day_time_meta_path)
    else:
        day_time_df = None

    # Normalize input filters
    if isinstance(stim_label, (list, np.ndarray)):
        stim_label = set(stim_label)
    if isinstance(age, (list, np.ndarray)):
        age = set(age)
    if isinstance(stim_type, str):
        stim_type = {stim_type}
    elif stim_type is not None:
        stim_type = set(stim_type)

    # Iterate through all subjects/trials in .npz file
    i = 0
    while _safe_get_field(loaded, 'power', i) is not None:
        power = _safe_get_field(loaded, 'power', i)
        s_id = _safe_get_field(loaded, 'subject_id', i, int)
        t_id = _safe_get_field(loaded, 'trial_id', i, int)
        gend = _safe_get_field(loaded, 'gender', i, str)
        hand = _safe_get_field(loaded, 'handiness', i, str)
        ag = _safe_get_field(loaded, 'age', i, int)
        label = _safe_get_field(loaded, 'label', i, int)
        img = _safe_get_field(loaded, 'img', i)
        task_type = _safe_get_field(loaded, 'task_type', i, str)
        phase = _safe_get_field(loaded, 'phase', i)
            

        # Apply filters
        # Filter by daytime condition
        if day_time is not None:
            cond_row = day_time_df[
                (day_time_df["Subject_id"] == s_id) & (day_time_df["Trial_id"] == t_id)
            ]
            if cond_row.empty or cond_row["Condition"].values[0] != day_time:
                i += 1
                continue

        # Filter by stim_type
        if stim_type is not None and task_type not in stim_type:
            i += 1
            continue

        # Filter by stim_label
        if stim_label is not None:
            if isinstance(stim_label, set):
                if label not in stim_label:
                    i += 1
                    continue
            elif label != stim_label:
                i += 1
                continue

        # Filter by gender
        if gender is not None and gend != gender:
            i += 1
            continue

        # Filter by age
        if age is not None:
            if isinstance(age, set):
                if ag not in age:
                    i += 1
                    continue
            elif ag != age:
                i += 1
                continue

        # Filter by handiness
        if handiness is not None and hand != handiness:
            i += 1
            continue

        results_arr.append([power, phase, s_id, t_id, gend, hand, ag, label, img, task_type])
        i += 1

    return results_arr