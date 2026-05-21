import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.detection import DAOStarFinder


def detect_sources(image, threshold_sigma=5.0, fwhm_guess=3.0):
    """Detect point-like sources (trail centres) in a CCD image.

    Uses ``DAOStarFinder`` to find compact maxima above a sigma-clipped
    background threshold.  The returned positions serve as initial parameter
    guesses for the trail fitter.

    Parameters
    ----------
    image : numpy.ndarray
    threshold_sigma : float
        Detection threshold in units of the background RMS.
    fwhm_guess : float
        Approximate PSF FWHM used by the matched filter, in pixels.

    Returns
    -------
    astropy.table.Table with columns ``x``, ``y``, ``peak``, ``flux``.
    """
    _mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm_guess, threshold=threshold_sigma * std)
    sources = finder(image - median)

    empty = Table(
        names=["x", "y", "peak", "flux"],
        dtype=[float, float, float, float],
    )
    if sources is None or len(sources) == 0:
        return empty

    x_col = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    y_col = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"
    sources.rename_column(x_col, "x")
    sources.rename_column(y_col, "y")
    return sources[["x", "y", "peak", "flux"]]
