import numpy as np
from astropy.table import Table
from scipy.optimize import minimize

from .models import TrailModel


def fit_trail(image, x0, y0, angle_guess=0.0, length_guess=10.0, fwhm_guess=3.0):
    """Fit a TrailModel to an image patch centred near (x0, y0).

    Uses Nelder-Mead simplex minimisation of the sum-of-squared residuals
    between the parametric trail model and the data.

    Parameters
    ----------
    image : numpy.ndarray
        Full image or cutout.  Fitting functions operate on numpy arrays;
        callers are responsible for FITS I/O.
    x0, y0 : float
        Initial position estimate in pixels.
    angle_guess : float
        Initial angle in radians.
    length_guess : float
        Initial trail length in pixels.
    fwhm_guess : float
        Initial PSF FWHM in pixels.

    Returns
    -------
    TrailModel with best-fit parameters, or ``None`` if the fit fails.
    """
    flux_guess = float(image.sum())
    p0 = [x0, y0, angle_guess, length_guess, fwhm_guess, flux_guess]

    def objective(p):
        model = TrailModel(*p)
        return float(np.sum(model.residuals(image) ** 2))

    result = minimize(
        objective,
        p0,
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 0.05, "fatol": 1.0},
    )
    if not result.success:
        return None
    return TrailModel(*result.x)


def fit_image(image, sources, fwhm_guess=3.0):
    """Fit trails to all sources detected in an image.

    Parameters
    ----------
    image : numpy.ndarray
    sources : astropy.table.Table
        Output of :func:`startrail.detect.detect_sources`, containing at
        least columns ``x`` and ``y``.
    fwhm_guess : float

    Returns
    -------
    astropy.table.Table with columns ``x0``, ``y0``, ``angle``, ``length``,
    ``fwhm``, ``flux`` (one row per successfully fitted trail).
    """
    empty = Table(
        names=["x0", "y0", "angle", "length", "fwhm", "flux"],
        dtype=[float] * 6,
    )
    rows = []
    for row in sources:
        result = fit_trail(image, row["x"], row["y"], fwhm_guess=fwhm_guess)
        if result is not None:
            rows.append(result.as_dict())

    if not rows:
        return empty
    return Table(rows=rows)
