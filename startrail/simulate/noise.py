import numpy as np


def add_sky(image, sky_level, seed=None):
    """Add Poisson-distributed sky background to an image.

    Parameters
    ----------
    image : numpy.ndarray
        Existing image in counts.
    sky_level : float
        Mean sky counts per pixel.
    seed : int or None

    Returns
    -------
    numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    return image + rng.poisson(sky_level, size=image.shape).astype(float)


def add_read_noise(image, read_noise, seed=None):
    """Add Gaussian read noise to an image.

    Parameters
    ----------
    image : numpy.ndarray
    read_noise : float
        RMS read noise in counts.
    seed : int or None

    Returns
    -------
    numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    return image + rng.normal(0.0, read_noise, size=image.shape)


def add_bias(image, bias_level):
    """Add a constant bias level to an image.

    Parameters
    ----------
    image : numpy.ndarray
    bias_level : float

    Returns
    -------
    numpy.ndarray
    """
    return image + bias_level
