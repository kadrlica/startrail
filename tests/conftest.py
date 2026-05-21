import numpy as np
import pytest
from astropy.table import Table

from startrail.simulate import make_image

SHAPE = (256, 256)


@pytest.fixture
def single_trail_stars():
    return Table(
        {
            "x": [128.0],
            "y": [128.0],
            "angle": [0.0],
            "length": [20.0],
            "flux": [50000.0],
        }
    )


@pytest.fixture
def simulated_image(single_trail_stars):
    return make_image(
        SHAPE,
        single_trail_stars,
        fwhm=3.0,
        sky_level=100.0,
        read_noise=5.0,
        bias=1000.0,
        seed=42,
    )


@pytest.fixture
def blank_image():
    return np.zeros(SHAPE)
