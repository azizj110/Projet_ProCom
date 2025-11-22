from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import mne
from braindecode.datasets.base import BaseConcatDataset, BaseDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)

from eegdash.dataset import DS003825

# --------------------------------------------------------------------------- #
#                               CONFIGURATION                                 #
# --------------------------------------------------------------------------- #

CACHE_DIR = Path("./data")
DATASET_ROOT = CACHE_DIR / "ds003825"
RESAMPLE_SFREQ = 100.0
WINDOW_DURATION_S = 0.5  # 500 ms windows after each stimulus onset
SUBJECTS = ["sub-02", "sub-03", "sub-04", "sub-05",
"sub-07","sub-08", "sub-09", "sub-10", "sub-11",
"sub-12", "sub-13", "sub-14", "sub-15", "sub-16", "sub-17",
"sub-18", "sub-20", "sub-21", "sub-22", "sub-23",
"sub-25", "sub-26", "sub-27", "sub-28", "sub-29",
"sub-30", "sub-31", "sub-32", "sub-33", "sub-34", "sub-35",
"sub-36", "sub-37", "sub-38", "sub-39", "sub-40", "sub-41",
"sub-42", "sub-43", "sub-44", "sub-45", "sub-46", "sub-47",
"sub-48"]

def build_object_mapping(dataset_root: Path, subjects: list[str] | None = None) -> dict[str, int]:
    """Collect a stable object -> integer mapping from events TSV files."""
    if subjects:
        event_files = [
            dataset_root / subj / "eeg" / f"{subj}_task-rsvp_events.tsv"
            for subj in subjects
        ]
    else:
        event_files = sorted(dataset_root.glob("sub-*/eeg/*_events.tsv"))

    mapping: OrderedDict[str, int] = OrderedDict()
    missing = []
    for tsv_path in event_files:
        if not tsv_path.exists():
            missing.append(tsv_path)
            continue
        df = pd.read_csv(tsv_path, sep="\t")
        if "object" not in df.columns:
            continue
        for obj in df["object"].dropna().astype(str):
            mapping.setdefault(obj, len(mapping))

    if not mapping:
        msg = "No object labels were found across events.tsv files"
        if missing:
            msg += f". Missing files: {missing[:3]}{'...' if len(missing) > 3 else ''}"
        raise RuntimeError(msg)
    return mapping


def annotate_raw_with_objects(base_dataset, window_duration: float) -> None:
    """Replace raw annotations with window segments labelled by the object name."""
    raw = base_dataset.raw
    bids_path = base_dataset.bidspath
    events_path = bids_path.copy().update(suffix="events", extension=".tsv").fpath

    df = pd.read_csv(events_path, sep="\t")
    if "object" not in df.columns:
        raise RuntimeError(f"Missing 'object' column in {events_path}")

    df = df[df["object"].notna()].copy()
    if df.empty:
        raise RuntimeError(f"No labelled events found in {events_path}")

    descriptions = df["object"].astype(str).to_numpy()
    onset_sec = (
        df["time_stimon"].astype(float)
        if "time_stimon" in df.columns
        else df["onset"].astype(float)
        / float(base_dataset.record.get("sampling_frequency", raw.info["sfreq"]))
    ).to_numpy()

    durations = np.full_like(onset_sec, window_duration, dtype=float)
    end_time = raw.times[-1]
    keep = (onset_sec >= 0) & ((onset_sec + durations) <= end_time)
    if not np.any(keep):
        raise RuntimeError(f"All annotations fell outside recording bounds for {events_path}")

    annotations = mne.Annotations(
        onset=onset_sec[keep],
        duration=durations[keep],
        description=descriptions[keep],
    )
    raw.set_annotations(annotations)


def main() -> None:
    subject_dirs = SUBJECTS or None
    subject_query = (
        [s.split("-", 1)[-1] if s.startswith("sub-") else s for s in SUBJECTS]
        if SUBJECTS
        else None
    )
    object_mapping = build_object_mapping(DATASET_ROOT, subject_dirs)
    dataset_kwargs = {"subject": subject_query} if subject_query else {}
    dataset = DS003825(cache_dir=CACHE_DIR, **dataset_kwargs)

    preprocess(
        concat_ds=dataset,
        preprocessors=[
            Preprocessor("load_data"),
            Preprocessor("resample", sfreq=RESAMPLE_SFREQ),
            Preprocessor("filter", l_freq=1.0, h_freq=40.0),
            # Preprocessor("pick_channels", ch_names=common_channels, ordered=True),
            #  Preprocessor(lambda raw: raw.set_eeg_reference("average")),
        ],
    )

    for base_ds in dataset.datasets:
        annotate_raw_with_objects(base_ds, window_duration=WINDOW_DURATION_S)

    sfreq = dataset.datasets[0].raw.info["sfreq"]
    window_samples = int(round(WINDOW_DURATION_S * sfreq))

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_samples,
        window_stride_samples=window_samples,
        mapping=object_mapping,
        preload=True,
        on_missing="ignore",
        accepted_bads_ratio=1.0,
    )

    subject_ids = [ds.record.get("subject") for ds in dataset.datasets]
    windows_by_subject = {
        subj: win_ds for subj, win_ds in zip(subject_ids, windows_dataset.datasets)
    }


    # windows_list : liste de WindowsDataset
    with open("windows_by_subject.pkl", "wb") as f:
        pickle.dump(windows_by_subject, f)
    with open("subject_ids.pkl", "wb") as f:
        pickle.dump(subject_ids, f)

if __name__ == "__main__":
    main()
