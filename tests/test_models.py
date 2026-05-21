import numpy as np
import pytest

from startrail.models import TrailModel

SHAPE = (64, 64)
MODEL = TrailModel(x0=32.0, y0=32.0, angle=0.0, length=10.0, fwhm=3.0, flux=1000.0)


def test_evaluate_shape():
    assert MODEL.evaluate(SHAPE).shape == SHAPE


def test_evaluate_flux():
    result = MODEL.evaluate(SHAPE)
    assert result.sum() == pytest.approx(1000.0, rel=0.02)


def test_residuals_zero_on_model():
    model_image = MODEL.evaluate(SHAPE)
    resid = MODEL.residuals(model_image)
    np.testing.assert_allclose(resid, 0.0, atol=1e-8)


def test_as_dict_keys():
    keys = set(MODEL.as_dict().keys())
    assert keys == {"x0", "y0", "angle", "length", "fwhm", "flux"}


def test_as_dict_values():
    d = MODEL.as_dict()
    assert d["x0"] == 32.0
    assert d["flux"] == 1000.0
