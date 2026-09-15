"""Download and read BCI Competition IV 2a GDF files."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import mne
import numpy as np

BASE_URL = "https://bnci-horizon-2020.eu/database/data-sets/001-2014"
CLASS_EVENT_CODES = {"769": "left_hand", "770": "right_hand", "771": "feet", "772": "tongue"}
CLASS_NAMES = tuple(CLASS_EVENT_CODES.values())


def download_subject(subject: int, root: str | Path, session: str = "T") -> Path:
    """Download one training (T) or evaluation (E) file if it is absent."""
    if not 1 <= subject <= 9:
        raise ValueError("subject must be in [1, 9]")
    session = session.upper()
    if session not in {"T", "E"}:
        raise ValueError("session must be 'T' or 'E'")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"A{subject:02d}{session}.gdf"
    if not path.exists():
        urlretrieve(f"{BASE_URL}/{path.name}", path)
    return path


def _find_event_id(raw: mne.io.BaseRaw) -> dict[str, int]:
    """Map the four task annotations, tolerating MNE's annotation naming."""
    annotations = raw.annotations
    found: dict[str, int] = {}
    for description in annotations.description:
        code = description.split("/")[-1]
        if code in CLASS_EVENT_CODES:
            found[CLASS_EVENT_CODES[code]] = int(code)
    missing = set(CLASS_NAMES) - set(found)
    if missing:
        raise RuntimeError(f"Could not find task annotations {sorted(missing)}")
    return found


def load_subject(path: str | Path, *, tmin: float = 0.5, tmax: float = 2.5,
                 l_freq: float = 4.0, h_freq: float = 38.0) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    """Read, filter, and epoch a subject's training GDF file.

    Returns ``(X, y, sfreq, channel_names)`` with X shaped ``(trials, channels, samples)``.
    Only the four motor-imagery event codes are retained; rejected/unknown events are ignored.
    """
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    # Keep EEG only and remove common non-EEG channels (EOG, trigger, etc.).
    raw.pick(picks="eeg")
    raw.filter(l_freq, h_freq, method="fir", phase="zero-double", verbose="ERROR")
    event_id = _find_event_id(raw)
    events, event_codes = mne.events_from_annotations(raw, event_id={str(v): v for v in event_id.values()}, verbose="ERROR")
    selected = np.isin(events[:, 2], list(event_id.values()))
    events = events[selected]
    code_to_label = {code: CLASS_NAMES.index(name) for name, code in event_id.items()}
    y = np.asarray([code_to_label[int(code)] for code in events[:, 2]], dtype=np.int64)
    epochs = mne.Epochs(raw, events, event_id={str(v): v for v in event_id.values()},
                        tmin=tmin, tmax=tmax, baseline=None, preload=True,
                        reject_by_annotation=True, verbose="ERROR")
    # Epoch rejection can remove trials; recover labels from the retained event codes.
    retained_codes = epochs.events[:, 2]
    y = np.asarray([code_to_label[int(code)] for code in retained_codes], dtype=np.int64)
    return epochs.get_data(copy=True).astype(np.float32), y, float(raw.info["sfreq"]), list(raw.ch_names)
