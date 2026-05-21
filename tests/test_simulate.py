import numpy as np
import pytest
from astropy.table import Table

from startrail.simulate import make_image
from startrail.simulate.noise import add_bias, add_read_noise, add_sky
from startrail.simulate.psf import gaussian_psf, moffat_psf
from startrail.simulate.trail import make_trail

SHAPE = (64, 64)


# --- PSF ---

def test_gaussian_psf_shape():
    assert gaussian_psf(21, fwhm=3.0).shape == (21, 21)


def test_gaussian_psf_normalized():
    assert abs(gaussian_psf(21, fwhm=3.0).sum() - 1.0) < 1e-10


def test_gaussian_psf_peak_at_centre():
    psf = gaussian_psf(21, fwhm=3.0)
    assert psf[10, 10] == psf.max()


def test_moffat_psf_normalized():
    assert abs(moffat_psf(21, fwhm=3.0).sum() - 1.0) < 1e-6


# --- Noise ---

def test_add_sky_mean():
    out = add_sky(np.zeros(SHAPE), sky_level=200.0, seed=0)
    assert out.mean() == pytest.approx(200.0, rel=0.05)


def test_add_read_noise_zero_mean():
    out = add_read_noise(np.zeros(SHAPE), read_noise=5.0, seed=0)
    assert abs(out.mean()) < 1.0


def test_add_bias_constant():
    out = add_bias(np.zeros(SHAPE), 1000.0)
    assert (out == 1000.0).all()


# --- Trail ---

def test_make_trail_flux():
    trail = make_trail(SHAPE, 32, 32, angle=0.0, length=10.0, fwhm=3.0, flux=5000.0)
    assert trail.sum() == pytest.approx(5000.0, rel=0.02)


def test_make_trail_shape():
    assert make_trail(SHAPE, 32, 32, angle=0.0, length=10.0, fwhm=3.0, flux=1.0).shape == SHAPE


def test_make_trail_peak_near_centre():
    trail = make_trail(SHAPE, 32, 32, angle=0.0, length=5.0, fwhm=3.0, flux=1000.0)
    cy, cx = np.unravel_index(trail.argmax(), trail.shape)
    assert abs(cx - 32) <= 2
    assert abs(cy - 32) <= 2


# --- Full image ---

def test_make_image_shape(simulated_image):
    assert simulated_image.shape == (256, 256)


def test_make_image_flux(simulated_image, single_trail_stars):
    ny, nx = simulated_image.shape
    background = (100.0 + 1000.0) * ny * nx
    trail_flux = simulated_image.sum() - background
    expected = float(single_trail_stars["flux"][0])
    assert trail_flux == pytest.approx(expected, rel=0.1)


def test_make_image_positive(simulated_image):
    assert simulated_image.min() > 0
