import numpy as np

from .trail import make_trail
from .noise import add_sky, add_read_noise, add_bias


def make_image(
    shape,
    stars,
    fwhm,
    sky_level=100.0,
    read_noise=5.0,
    bias=1000.0,
    jitter_sigma=0.0,
    tau=None,
    seed=None,
):
    """Simulate a CCD image containing open-shutter star trails.

    Parameters
    ----------
    shape : (ny, nx)
    stars : astropy.table.Table
        One row per star with columns ``x``, ``y``, ``angle``, ``length``,
        ``flux``.
    fwhm : float
        PSF FWHM in pixels (same for all stars).
    sky_level : float
        Mean sky background in counts per pixel.
    read_noise : float
        Read-noise RMS in counts.
    bias : float
        Constant bias level in counts.
    jitter_sigma : float
        Steady-state jitter RMS in pixels perpendicular to each trail.
    tau : float or None
        Guider timescale in pixels (see :func:`make_trail`).
    seed : int or None

    Returns
    -------
    numpy.ndarray of shape ``shape``
    """
    rng = np.random.default_rng(seed)
    image = np.zeros(shape)

    for star in stars:
        trail_seed = int(rng.integers(0, 2**31))
        image += make_trail(
            shape,
            x0=float(star["x"]),
            y0=float(star["y"]),
            angle=float(star["angle"]),
            length=float(star["length"]),
            fwhm=fwhm,
            flux=float(star["flux"]),
            jitter_sigma=jitter_sigma,
            tau=tau,
            seed=trail_seed,
        )

    image = add_sky(image, sky_level, seed=int(rng.integers(0, 2**31)))
    image = add_read_noise(image, read_noise, seed=int(rng.integers(0, 2**31)))
    image = add_bias(image, bias)
    return image
