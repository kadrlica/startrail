import numpy as np
import pytest
from astropy.table import Table

from startrail.io import read_catalog, read_image, write_catalog, write_image


def test_fits_image_roundtrip(tmp_path):
    data = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    path = tmp_path / "test.fits"
    write_image(path, data)
    data_back, header = read_image(path)
    assert data_back.shape == data.shape
    np.testing.assert_allclose(data_back, data, rtol=1e-5)


def test_fits_catalog_roundtrip(tmp_path):
    cat = Table({"x0": [1.0, 2.0], "y0": [3.0, 4.0], "flux": [100.0, 200.0]})
    path = tmp_path / "cat.fits"
    write_catalog(cat, path)
    cat_back = read_catalog(path)
    np.testing.assert_array_equal(cat_back["x0"], cat["x0"])
    np.testing.assert_array_equal(cat_back["flux"], cat["flux"])


def test_ecsv_catalog_roundtrip(tmp_path):
    cat = Table({"x0": [5.5], "flux": [999.0]})
    path = tmp_path / "cat.ecsv"
    write_catalog(cat, path)
    cat_back = read_catalog(path)
    np.testing.assert_allclose(cat_back["x0"], cat["x0"])
