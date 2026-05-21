import numpy as np
import pytest
from astropy.table import Table

from startrail.detect import detect_sources
from startrail.simulate import make_image

SHAPE = (128, 128)


def test_detect_finds_bright_source():
    stars = Table(
        {"x": [64.0], "y": [64.0], "angle": [0.0], "length": [5.0], "flux": [200000.0]}
    )
    image = make_image(SHAPE, stars, fwhm=3.0, sky_level=100.0, read_noise=5.0, bias=0.0, seed=0)
    sources = detect_sources(image, threshold_sigma=5.0, fwhm_guess=3.0)
    assert len(sources) >= 1


def test_detect_empty_image():
    image = np.ones(SHAPE) * 1000.0
    sources = detect_sources(image, threshold_sigma=5.0)
    assert len(sources) == 0


def test_detect_returns_correct_columns():
    image = np.ones(SHAPE) * 1000.0
    sources = detect_sources(image, threshold_sigma=5.0)
    assert set(sources.colnames) == {"x", "y", "peak", "flux"}
