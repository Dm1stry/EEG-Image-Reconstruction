"""
Statistical_Tester — module for automated hypothesis testing on EEG band-power data.

Builds band-power tables from NPZ spectra, runs group comparisons (e.g. t-test),
and produces topomap visualizations. Use export_band_table() to get a band table
from NPZ; use stat_test() to compare two groups and get statistics plus figures.
"""
from __future__ import annotations

import os
import re
import csv
import gc
import warnings
from typing import Dict, Optional, Any, Tuple, Iterable, List, Union

import numpy as np
import pandas as pd
from scipy import stats

import mne
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None

def _get_day_time_df(day_time_meta_path: str) -> pd.DataFrame:
    meta1 = pd.read_excel(day_time_meta_path, sheet_name=0, header=1).rename(columns={
        "Subject ID": "Subject_id",
        "Время начала записи": "Time",
    })
    meta1["Trial_id"] = 1

    meta2 = pd.read_excel(day_time_meta_path, sheet_name=1, header=1).rename(columns={
        "Subject ID": "Subject_id",
        "Время начала записи": "Time",
    })
    meta2["Trial_id"] = 2

    meta = pd.concat([meta1, meta2], ignore_index=True)

    meta["Subject_id"] = (
        meta["Subject_id"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype("float")
    )
    meta = meta[meta["Subject_id"].notna()].copy()
    meta["Subject_id"] = meta["Subject_id"].astype(int)

    s = meta["Time"].astype(str).str.strip()
    n = pd.to_numeric(s, errors="coerce")
    dt_str = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
    dt_num = pd.to_datetime(n, errors="coerce", origin="1899-12-30", unit="D")
    dt = dt_str.fillna(dt_num)

    meta["Hour"] = dt.dt.hour
    meta["Time"] = dt.dt.strftime("%H:%M")

    meta_idx = (
        meta[["Subject_id", "Trial_id", "Hour", "Time"]]
        .dropna(subset=["Hour"])
        .drop_duplicates(["Subject_id", "Trial_id"], keep="last")
    )

    meta_idx["Condition"] = pd.cut(
        meta_idx["Hour"],
        bins=[-0.1, 10, 18, 24],
        labels=["Other", "Day", "Evening"],
        right=False,
        include_lowest=True,
    ).astype(object).fillna("Other")

    return meta_idx

FIELD_SYNONYMS = {
    "power":      ["power", "pwr", "psd"],
    "phase":      ["phase", "phs", "phse"],
    "subject_id": ["subject_id", "sid", "s_id", "subj_id"],
    "trial_id":   ["trial_id", "tid", "t_id"],
    "gender":     ["gender", "gend", "gen"],
    "handiness":  ["handiness", "hand"],
    "age":        ["age", "ag"],
    "label":      ["label", "true_label", "labl", "lbl"],
    "img":        ["img", "image", "picture", "pattern", "patrn", "pictr"],
    "task_type":  ["task_type", "task", "pattern_type", "p_type", "image_type"],
}


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _build_npz_key_index(npz_files: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in npz_files:
        out[_normalize_name(k)] = k
    return out


def _get_field(loaded: dict, key_index: Dict[str, str], base_field: str, i: int):
    candidates: List[str] = []
    for synonym in FIELD_SYNONYMS.get(base_field, []):
        candidates += [
            f"{synonym}_{i}",
            f"{synonym}{i}",
            f"{synonym}-{i}",
            f"{synonym}{{{i}}}",
            synonym,
        ]
    for cand in candidates:
        nk = _normalize_name(cand)
        if nk in key_index:
            return loaded[key_index[nk]]
    return None


def _safe_get_field(loaded, key_index, field, i, cast=None):
    try:
        val = _get_field(loaded, key_index, field, i)
        if val is None:
            return None
        return cast(val) if cast else val
    except Exception:
        return None


def _count_trials_in_npz(npz_files: List[str]) -> Optional[int]:
    syns = [ _normalize_name(s) for s in FIELD_SYNONYMS.get("power", []) ]
    idxs = set()
    for k in npz_files:
        nk = _normalize_name(k)
        for s in syns:
            m = re.match(rf"^{s}(\d+)$", nk)
            if m:
                idxs.add(int(m.group(1)))
    return len(idxs) if idxs else None

DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "Delta": (1, 4),
    "Tetta": (4, 7),
    "Alpha": (7, 13),
    "Beta":  (13, 30),
}


def make_band_cols_from_bands(
    bands: Dict[str, Tuple[float, float]],
    freqs: np.ndarray,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for band_name, (f_lo, f_hi) in bands.items():
        out[band_name] = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
    return out


def _band_means_per_channel(power: np.ndarray, band_cols: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    arr = np.asarray(power)
    if arr.ndim == 3:
        arr = np.nanmean(arr, axis=2)  # (C,F)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected power shape: {arr.shape}")

    out: Dict[str, np.ndarray] = {}
    for band, idx in band_cols.items():
        if idx.size == 0 or np.max(idx) >= arr.shape[1]:
            out[band] = np.full(arr.shape[0], np.nan, float)
        else:
            out[band] = np.nanmean(arr[:, idx], axis=1)
    return out


def build_band_table_from_npz(
    npz_path: str,
    out_csv: Optional[str] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    *,
    day_time_meta_path: Optional[str] = None,
    flush_rows: int = 200,
    freq_min: float = 2.0,   # как в ноутбуке
    freq_max: float = 40.0,  # как в ноутбуке
    progress: bool = True,
) -> pd.DataFrame:

    if bands is None:
        bands = DEFAULT_BANDS.copy()

    dt_map = None
    if day_time_meta_path is not None:
        dt_df = _get_day_time_df(day_time_meta_path)
        dt_map = {(int(r.Subject_id), int(r.Trial_id)): str(r.Condition) for _, r in dt_df.iterrows()}

    with np.load(npz_path, allow_pickle=True) as loaded:
        key_index = _build_npz_key_index(list(loaded.files))

        # пример power
        i0 = 0
        ex = _safe_get_field(loaded, key_index, "power", i0)
        while ex is None:
            i0 += 1
            ex = _safe_get_field(loaded, key_index, "power", i0)
            if i0 > 10000:
                raise RuntimeError("Не нашёл power_* по индексной схеме i=0..10000.")

        ex = np.asarray(ex)
        if ex.ndim == 3:
            n_channels, n_freqs, _ = ex.shape
        elif ex.ndim == 2:
            n_channels, n_freqs = ex.shape
        else:
            raise RuntimeError(f"Неожиданная форма power: {ex.shape}")

        freqs = np.linspace(float(freq_min), float(freq_max), int(n_freqs))
        band_cols = make_band_cols_from_bands(bands, freqs)
        band_names = list(band_cols.keys())

        feat_cols = [f"ch{ch}_{bn}" for ch in range(int(n_channels)) for bn in band_names]
        meta_cols = [
            "s_id", "t_id", "gender", "handiness", "age",
            "stim_label", "label", "img", "task_type", "stim_type", "day_time",
        ]
        header = meta_cols + feat_cols

        total = _count_trials_in_npz(list(loaded.files)) if progress else None
        pbar = _tqdm(total=total, desc="Export band_table", unit="trial") if (progress and _tqdm is not None) else None

        buffer: List[Dict[str, Any]] = []

        i = 0
        while True:
            power = _safe_get_field(loaded, key_index, "power", i)
            if power is None:
                break

            power = np.asarray(power)
            if power.ndim not in (2, 3):
                i += 1
                continue

            s_id = _safe_get_field(loaded, key_index, "subject_id", i, int)
            t_id = _safe_get_field(loaded, key_index, "trial_id", i, int)
            gender = _safe_get_field(loaded, key_index, "gender", i, str)
            handiness = _safe_get_field(loaded, key_index, "handiness", i, str)
            age = _safe_get_field(loaded, key_index, "age", i, int)
            label = _safe_get_field(loaded, key_index, "label", i, int)
            img = _safe_get_field(loaded, key_index, "img", i)
            task_type = _safe_get_field(loaded, key_index, "task_type", i, str)

            row: Dict[str, Any] = {
                "s_id": s_id,
                "t_id": t_id,
                "gender": gender,
                "handiness": handiness,
                "age": age,
                "stim_label": label,
                "label": label,
                "img": img,
                "task_type": task_type,
                "stim_type": task_type,
            }

            if dt_map is not None and s_id is not None and t_id is not None:
                row["day_time"] = dt_map.get((int(s_id), int(t_id)), np.nan)
            else:
                row["day_time"] = np.nan

            band_means = _band_means_per_channel(power, band_cols)
            for ch in range(int(n_channels)):
                for bn in band_names:
                    row[f"ch{ch}_{bn}"] = float(band_means[bn][ch]) if ch < len(band_means[bn]) else np.nan

            buffer.append(row)

            i += 1
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    df = pd.DataFrame(buffer)
    if out_csv is not None:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df



def export_band_table(
    npz_path: str,
    out_csv: Optional[str] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    day_time_meta_path: Optional[str] = None,
    flush_rows: int = 200,
) -> pd.DataFrame:
    """
    Build a band-power table from an NPZ file and return it as a DataFrame.

    Reads power spectra from the NPZ, computes mean power per channel per band
    (Delta, Theta, Alpha, Beta by default), and returns a table with metadata
    (s_id, t_id, gender, etc.) and feature columns (ch0_Delta, ch0_Alpha, ...).
    Optionally saves the table to CSV only when out_csv is provided.

    Parameters
    ----------
    npz_path : str
        Path to the NPZ file with power (and optional metadata) arrays.
    out_csv : str, optional
        If given, the table is saved to this CSV path. If omitted, nothing is saved.
    bands : dict, optional
        Frequency bands as {name: (low_hz, high_hz)}. Defaults to Delta, Theta, Alpha, Beta.
    day_time_meta_path : str, optional
        Path to Excel with day/time metadata to add a "day_time" column.
    flush_rows : int
        Unused; kept for API compatibility.

    Returns
    -------
    pd.DataFrame
        The band-power table (one row per trial).
    """
    return build_band_table_from_npz(
        npz_path=npz_path,
        out_csv=out_csv,
        bands=bands,
        day_time_meta_path=day_time_meta_path,
        flush_rows=flush_rows,
        freq_min=2.0,
        freq_max=40.0,
        progress=True,
    )

def load_prepared_eeg(
    fif_path: str,
    *,
    montage: str = "standard_1020",
    preload: bool = True,
    verbose: bool = False,
) -> mne.io.BaseRaw:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw = mne.io.read_raw_fif(fif_path, preload=preload, verbose=verbose)

    eeg = raw.copy().pick_types(eeg=True)
    eeg.set_montage(montage, on_missing="warn", match_case=False)
    return eeg

_CH_BAND_RE = re.compile(r"^ch(\d+)_(.+)$")


def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if isinstance(c, str) and _CH_BAND_RE.match(c)]


def parse_ch_band(col: str) -> tuple[Optional[int], Optional[str]]:
    m = _CH_BAND_RE.match(col)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def infer_bands(df: pd.DataFrame) -> list[str]:
    bands = set()
    for c in infer_feature_cols(df):
        _, band = parse_ch_band(c)
        if band:
            bands.add(band)
    return sorted(bands)


def load_band_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    for c in ("s_id", "t_id", "age", "label", "stim_label"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("gender", "handiness", "day_time", "stim_type", "task_type", "img"):
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df


def _normalize_value(v: Any) -> Any:
    if pd.isna(v):
        return v
    if isinstance(v, str):
        return v.strip()
    return v


def filter_df_by_group_spec(
    df: pd.DataFrame,
    group_spec: Dict[str, Any],
    feature_groups: Optional[Dict[str, Dict[str, list[Any]]]] = None,
) -> pd.DataFrame:
    """
    Фильтрует df по спецификации группы.

    group_spec:
      - простой вариант: {"gender":"m", "day_time":"Day"}
      - пересечение:    {"day_time":"Day", "gender":"m", "handiness":"r"}

    Если feature_groups задан (ваш FEATURE_GROUPS), то можно писать:
      {"age":"18-22"} или {"stim_label":"figure"} — модуль сам подставит allowed values.

    ВАЖНО: если в df есть колонка "stim_label", то фильтр stim_label идёт по ней,
           иначе — по "label".
    """
    out = df

    for feat, group_or_value in group_spec.items():
        if feat == "stim_label":
            col = "stim_label" if "stim_label" in out.columns else ("label" if "label" in out.columns else "stim_label")
        elif feat == "stim_type" and "stim_type" not in out.columns and "task_type" in out.columns:
            col = "task_type"  # fallback
        else:
            col = feat

        if col not in out.columns:
            raise KeyError(f"В band_table нет колонки '{col}' (из spec '{feat}')")

        series = out[col].map(_normalize_value)

        if feature_groups is not None and feat in feature_groups and isinstance(group_or_value, str):
            groups_dict = feature_groups[feat]
            if group_or_value not in groups_dict:
                raise KeyError(f"Нет группы '{group_or_value}' в FEATURE_GROUPS['{feat}']")
            allowed = set(groups_dict[group_or_value])

            if pd.api.types.is_numeric_dtype(series):
                mask = series.isin(list(allowed))
            else:
                mask = series.astype("string").isin([str(x) for x in allowed])
        else:
            if isinstance(group_or_value, (list, tuple, set, np.ndarray, pd.Series)):
                allowed = set([_normalize_value(x) for x in group_or_value])
                if pd.api.types.is_numeric_dtype(series):
                    mask = series.isin(list(allowed))
                else:
                    mask = series.astype("string").isin([str(x) for x in allowed])
            else:
                val = _normalize_value(group_or_value)
                if pd.api.types.is_numeric_dtype(series):
                    mask = series == val
                else:
                    mask = series.astype("string") == str(val)

        out = out[mask].copy()

    return out


def _aggregate_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    level = str(level).lower().strip()
    if level not in ("trial", "subject"):
        raise ValueError("level must be 'trial' or 'subject'")
    if level == "trial":
        return df

    if "s_id" not in df.columns:
        raise KeyError("Для level='subject' нужна колонка 's_id'")

    feat_cols = infer_feature_cols(df)
    meta_cols = [c for c in df.columns if c not in feat_cols]

    agg = {c: "first" for c in meta_cols if c != "s_id"}
    for c in feat_cols:
        agg[c] = "mean"

    return df.groupby("s_id", dropna=True, sort=False).agg(agg).reset_index()


def compute_ttest_stats_from_band_table(
    band_table: Union[str, pd.DataFrame],
    groupA: Dict[str, Any],
    groupB: Dict[str, Any],
    *,
    feature_groups: Optional[Dict[str, Dict[str, list[Any]]]] = None,
    bands: Optional[Iterable[str]] = None,
    level: str = "subject",
    equal_var: bool = False,   # Welch по умолчанию
    nan_policy: str = "omit",
    min_n: int = 5,
) -> pd.DataFrame:
    """
    Welch t-test для каждой пары channel-band между groupA и groupB.

    Возвращает DataFrame:
      channel, band, n_A, n_B, mean_A, mean_B, diff_A_minus_B, t_stat, p_value
    """
    df = load_band_table(band_table) if isinstance(band_table, str) else band_table

    A = filter_df_by_group_spec(df, groupA, feature_groups=feature_groups)
    B = filter_df_by_group_spec(df, groupB, feature_groups=feature_groups)

    A = _aggregate_level(A, level=level)
    B = _aggregate_level(B, level=level)

    feat_cols = infer_feature_cols(df)
    if bands is None:
        bands = infer_bands(df)
    bands = list(bands)

    rows = []
    for col in feat_cols:
        ch, band = parse_ch_band(col)
        if ch is None or band is None or band not in bands:
            continue

        x = pd.to_numeric(A[col], errors="coerce").to_numpy()
        y = pd.to_numeric(B[col], errors="coerce").to_numpy()
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        if x.size < min_n or y.size < min_n:
            continue

        try:
            t, p = stats.ttest_ind(x, y, equal_var=equal_var, nan_policy=nan_policy)
        except Exception:
            t, p = np.nan, np.nan

        rows.append({
            "channel": int(ch),
            "band": str(band),
            "n_A": int(x.size),
            "n_B": int(y.size),
            "mean_A": float(np.nanmean(x)),
            "mean_B": float(np.nanmean(y)),
            "diff_A_minus_B": float(np.nanmean(x) - np.nanmean(y)),
            "t_stat": float(t) if np.isfinite(t) else np.nan,
            "p_value": float(p) if np.isfinite(p) else np.nan,
        })

    out = pd.DataFrame(
    rows,
    columns=[
        "channel","band","n_A","n_B",
        "mean_A","mean_B","diff_A_minus_B",
        "t_stat","p_value"
    ],
)
    if not out.empty:
        out = out.sort_values(["p_value", "channel", "band"], ascending=[True, True, True]).reset_index(drop=True)
    return out

def _vector_from_stats(stats_table, band, info, value_col):
    n_channels = len(info["ch_names"])
    vec = np.full(n_channels, np.nan, float)

    if stats_table is None or not isinstance(stats_table, pd.DataFrame):
        return vec
    if stats_table.empty:
        return vec
    if "band" not in stats_table.columns or "channel" not in stats_table.columns or value_col not in stats_table.columns:
        return vec

    df = stats_table[stats_table["band"] == band]
    for _, r in df.iterrows():
        ch = int(r["channel"])
        if 0 <= ch < n_channels:
            vec[ch] = float(r[value_col])
    return vec



def _vector_mean_from_group(df_group: pd.DataFrame, band: str, info) -> np.ndarray:
    n_channels = len(info["ch_names"])
    vec = np.full(n_channels, np.nan, float)

    cols = [c for c in df_group.columns if c.endswith(f"_{band}") and c.startswith("ch")]
    if not cols:
        return vec
    num = df_group[cols].apply(pd.to_numeric, errors="coerce")
    means = num.mean(axis=0, skipna=True).to_numpy()

    for j, c in enumerate(cols):
        ch, _ = parse_ch_band(c)
        if ch is not None and 0 <= ch < n_channels:
            vec[ch] = means[j]
    return vec



def stat_test(
    band_table: pd.DataFrame,
    groupA: Dict[str, Any],
    groupB: Dict[str, Any],
    *,
    example_eeg: str,
    test_type: str = "t-test",
    feature_groups: Optional[Dict[str, Dict[str, list[Any]]]] = None,
    bands: Optional[Iterable[str]] = None,
    level: str = "subject",
    alpha_mask: float = 0.05,
    title_prefix: str = "",
    show: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, plt.Figure]]:
    """
    Run a group comparison (e.g. t-test) and plot topomaps per band.

    For each band, plots mean(A), mean(B), A−B, t-stat, and -log10(p) with
    significance markers. Returns the statistics table and a dict of figures.

    Parameters
    ----------
    band_table : pd.DataFrame
        Band-power table (e.g. from export_band_table). Must contain meta columns
        and ch*_BandName feature columns.
    groupA, groupB : dict
        Group specs for filtering (e.g. {"day_time": "Day"}, {"gender": "f"}).
    example_eeg : str
        Path to a FIF file used to get channel layout/montage for topomaps.
    test_type : str
        Type of test. Only "t-test" is implemented; TODO: add other tests
        (e.g. Mann-Whitney, permutation).
    feature_groups : dict, optional
        Mapping of feature names to allowed group values (for group_spec).
    bands : iterable of str, optional
        Band names to include. Inferred from band_table if omitted.
    level : str
        Aggregation level: "trial" or "subject".
    alpha_mask : float
        p-value threshold for significance markers on the -log10(p) topomap.
    title_prefix : str
        Prefix for figure titles.
    show : bool
        Whether to call plt.show() for each figure.

    Returns
    -------
    stats_table : pd.DataFrame
        Per channel/band statistics (t_stat, p_value, mean_A, mean_B, etc.).
    figs : dict
        {band_name: matplotlib Figure} for each band.
    """
    # TODO: add other test_type implementations (e.g. Mann-Whitney, permutation tests)
    if test_type != "t-test":
        raise ValueError(f"test_type must be 't-test'; got {test_type!r}. TODO: implement other tests.")

    eeg = load_prepared_eeg(example_eeg)
    info = eeg.info

    df = band_table
    A = filter_df_by_group_spec(df, groupA, feature_groups=feature_groups)
    B = filter_df_by_group_spec(df, groupB, feature_groups=feature_groups)

    A = _aggregate_level(A, level=level)
    B = _aggregate_level(B, level=level)

    if bands is None:
        bands = infer_bands(df)
    bands = list(bands)

    stats_table = compute_ttest_stats_from_band_table(
        band_table,
        groupA=groupA,
        groupB=groupB,
        feature_groups=feature_groups,
        bands=bands,
        level=level,
    )
    if stats_table.empty:
        raise ValueError(
        "stats_table пустая: после фильтрации/агрегации слишком мало данных. "
        "Проверь значения day_time/gender в band_table.csv и/или снизь min_n, "
        "или попробуй level='trial'."
    )

    figs: Dict[str, plt.Figure] = {}
    for band in bands:
        meanA = _vector_mean_from_group(A, band, info)
        meanB = _vector_mean_from_group(B, band, info)
        diff = meanA - meanB

        tvals = _vector_from_stats(stats_table, band, info, "t_stat")
        pvals = _vector_from_stats(stats_table, band, info, "p_value")

        # limits
        vmin_ab = float(np.nanmin([np.nanmin(meanA), np.nanmin(meanB)]))
        vmax_ab = float(np.nanmax([np.nanmax(meanA), np.nanmax(meanB)]))
        if not np.isfinite(vmin_ab) or not np.isfinite(vmax_ab) or vmin_ab == vmax_ab:
            vlim_ab = (-1.0, 1.0)
        else:
            vlim_ab = (vmin_ab, vmax_ab)

        vmax_diff = float(np.nanmax(np.abs(diff))) if np.isfinite(np.nanmax(np.abs(diff))) else 1.0
        vlim_diff = (-vmax_diff, vmax_diff) if vmax_diff > 0 else (-1.0, 1.0)

        vmax_t = float(np.nanmax(np.abs(tvals))) if np.isfinite(np.nanmax(np.abs(tvals))) else 1.0
        vlim_t = (-vmax_t, vmax_t) if vmax_t > 0 else (-1.0, 1.0)

        # p map = -log10(p)
        p_plot = -np.log10(np.clip(pvals, 1e-300, 1.0))
        vmax_p = float(np.nanmax(p_plot)) if np.isfinite(np.nanmax(p_plot)) else 1.0
        vlim_p = (0.0, vmax_p if vmax_p > 0 else 1.0)

        mask_sig = np.isfinite(pvals) & (pvals <= float(alpha_mask))
        mask_params = dict(
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgewidth=2,
            markeredgecolor="black",
        )

        fig, axs = plt.subplots(1, 5, figsize=(18, 4.2))
        mne.viz.plot_topomap(meanA, info, axes=axs[0], show=False, contours=0, vlim=vlim_ab)
        axs[0].set_title(f"{band}\nmean(A)")

        mne.viz.plot_topomap(meanB, info, axes=axs[1], show=False, contours=0, vlim=vlim_ab)
        axs[1].set_title(f"{band}\nmean(B)")

        mne.viz.plot_topomap(diff, info, axes=axs[2], show=False, contours=0, vlim=vlim_diff, cmap="RdBu_r")
        axs[2].set_title(f"{band}\nA − B")

        mne.viz.plot_topomap(tvals, info, axes=axs[3], show=False, contours=0, vlim=vlim_t, cmap="RdBu_r")
        axs[3].set_title(f"{band}\nt-stat")

        mne.viz.plot_topomap(p_plot, info, axes=axs[4], show=False, contours=0, vlim=vlim_p,
                             mask=mask_sig, mask_params=mask_params)
        axs[4].set_title(f"{band}\n-log10(p)")

        if title_prefix:
            fig.suptitle(f"{title_prefix} — {band}", y=1.05)

        plt.tight_layout()
        if show:
            plt.show()

        figs[band] = fig

    return stats_table, figs
