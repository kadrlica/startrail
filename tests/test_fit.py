import numpy as np
import pytest
from astropy.table import Table

from startrail.fit import fit_image, fit_trail
from startrail.simulate import make_image

SHAPE = (64, 64)
TRUE = dict(x0=32.0, y0=32.0, angle=0.0, length=10.0, fwhm=3.0, flux=50000.0)


@pytest.fixture
def noiseless_trail():
    stars = Table(
        {
            "x": [TRUE["x0"]],
            "y": [TRUE["y0"]],
            "angle": [TRUE["angle"]],
            "length": [TRUE["length"]],
            "flux": [TRUE["flux"]],
        }
    )
    return make_image(
        SHAPE, stars, fwhm=TRUE["fwhm"], sky_level=0.0, read_noise=0.0, bias=0.0, seed=0
    )


def test_fit_trail_returns_model(noiseless_trail):
    result = fit_trail(noiseless_trail, x0=30.0, y0=30.0, fwhm_guess=3.0, length_guess=10.0)
    assert result is not None


def test_fit_trail_position_accuracy(noiseless_trail):
    result = fit_trail(noiseless_trail, x0=30.0, y0=30.0, fwhm_guess=3.0, length_guess=10.0)
    assert result is not None
    assert abs(result.x0 - TRUE["x0"]) < 2.0
    assert abs(result.y0 - TRUE["y0"]) < 2.0


def test_fit_image_returns_table(noiseless_trail):
    sources = Table({"x": [32.0], "y": [32.0], "peak": [1000.0], "flux": [50000.0]})
    result = fit_image(noiseless_trail, sources)
    assert isinstance(result, Table)
    assert len(result) == 1
    assert "x0" in result.colnames
