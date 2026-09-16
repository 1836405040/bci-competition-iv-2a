import numpy as np
import mne
from zipfile import ZipFile

from bci_2a import data
from bci_2a.preprocessing import ChannelwiseScaler, make_csp_pipeline


def test_channelwise_scaler_shape_and_centering():
    X = np.random.default_rng(0).normal(size=(8, 3, 20)).astype("float32")
    transformed = ChannelwiseScaler().fit_transform(X)
    assert transformed.shape == X.shape
    assert np.allclose(transformed.mean(axis=(0, 2)), 0, atol=1e-6)


def test_csp_pipeline_builds():
    assert make_csp_pipeline("lda").named_steps["csp"].n_components == 6


def test_download_subject_extracts_cached_archive(tmp_path):
    archive = tmp_path / data.ARCHIVE_NAME
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("BCICIV_2a_gdf/A01T.gdf", b"gdf-content")
    path = data.download_subject(1, tmp_path)
    assert path.read_bytes() == b"gdf-content"


def test_eog_channels_are_not_part_of_competition_eeg_set():
    names = [f"EEG-{index}" for index in range(22)] + ["EOG-left", "EOG-right", "EOG-central"]
    info = mne.create_info(names, 250, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((25, 10)), info, verbose="ERROR")
    eog = [name for name in raw.ch_names if "EOG" in name.upper()]
    raw.drop_channels(eog).pick("eeg")
    assert len(raw.ch_names) == 22
