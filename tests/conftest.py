import numpy as np
import pytest

from pulse_qc.synth import synthetic_ppg


@pytest.fixture(scope="session")
def clean():
    return synthetic_ppg(duration_s=60, fs=125, hr_bpm=72, seed=1)


@pytest.fixture
def x(clean) -> np.ndarray:
    return clean.signal.copy()
