import numpy as np


def gaussian_psf(size, fwhm):
    """Return a normalized 2-D Gaussian PSF array.

    Parameters
    ----------
    size : int
        Side length of the output array (should be odd).
    fwhm : float
        Full-width at half-maximum in pixels.

    Returns
    -------
    numpy.ndarray of shape (size, size), normalized to sum == 1.
    """
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    c = size // 2
    y, x = np.mgrid[-c : c + 1, -c : c + 1]
    psf = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return psf / psf.sum()


def moffat_psf(size, fwhm, beta=4.0):
    """Return a normalized 2-D Moffat PSF array.

    Parameters
    ----------
    size : int
        Side length of the output array (should be odd).
    fwhm : float
        Full-width at half-maximum in pixels.
    beta : float
        Moffat power-law index (default 4).

    Returns
    -------
    numpy.ndarray of shape (size, size), normalized to sum == 1.
    """
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    c = size // 2
    y, x = np.mgrid[-c : c + 1, -c : c + 1]
    r2 = x**2 + y**2
    psf = (1.0 + r2 / alpha**2) ** (-beta)
    return psf / psf.sum()
