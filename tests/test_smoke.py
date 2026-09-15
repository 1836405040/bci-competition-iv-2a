import numpy as np

from bci_2a.preprocessing import ChannelwiseScaler, make_csp_pipeline


def test_channelwise_scaler_shape_and_centering():
    X = np.random.default_rng(0).normal(size=(8, 3, 20)).astype("float32")
    transformed = ChannelwiseScaler().fit_transform(X)
    assert transformed.shape == X.shape
    assert np.allclose(transformed.mean(axis=(0, 2)), 0, atol=1e-6)


def test_csp_pipeline_builds():
    assert make_csp_pipeline("lda").named_steps["csp"].n_components == 6
