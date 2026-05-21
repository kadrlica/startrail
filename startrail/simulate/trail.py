import numpy as np

from ..utils import fwhm_to_sigma


def make_trail(
    shape, x0, y0, angle, length, fwhm, flux,
    jitter_sigma=0.0, tau=None, seed=None,
):
    """Generate a star trail image.

    The trail is modelled as the integral of a Gaussian PSF along the star's
    path during CCD readout.  Jitter is perpendicular to the nominal trail
    direction and follows an Ornstein-Uhlenbeck process when ``tau`` is given,
    modelling a telescope guider that provides a restoring force back toward
    the nominal pointing on a characteristic timescale ``tau`` (pixels along
    the trail).  When ``tau`` is ``None`` the jitter is a pure random walk.

    Parameters
    ----------
    shape : (ny, nx)
    x0, y0 : float
        Trail centre in pixel coordinates (column, row).
    angle : float
        Trail angle in radians.  0 = along the y / readout axis.
    length : float
        Trail length in pixels.
    fwhm : float
        PSF FWHM in pixels.
    flux : float
        Total trail flux in counts.
    jitter_sigma : float
        Steady-state RMS jitter amplitude in pixels perpendicular to the
        trail.  For the OU model this is the equilibrium standard deviation;
        for the random-walk model it is the per-unit-length diffusion
        coefficient.
    tau : float or None
        Guider timescale in pixels along the trail.  The star is pulled back
        toward its nominal position with an e-folding length of ``tau``.
        ``None`` disables the restoring force (pure random walk).
    seed : int or None

    Returns
    -------
    numpy.ndarray of shape ``shape``
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    sigma = fwhm_to_sigma(fwhm)

    n_steps = max(int(length * 3), 30)
    t_vals = np.linspace(-length / 2.0, length / 2.0, n_steps)

    sin_a = np.sin(angle)
    cos_a = np.cos(angle)

    # Perpendicular jitter
    if jitter_sigma > 0.0 and n_steps > 1:
        dt = abs(t_vals[1] - t_vals[0])
        jitter = np.zeros(n_steps)
        if tau is not None:
            # Ornstein-Uhlenbeck: exact discretisation with e-folding length tau.
            # Steady-state variance == jitter_sigma^2.
            decay = np.exp(-dt / tau)
            drive = jitter_sigma * np.sqrt(1.0 - decay ** 2)
            for i in range(1, n_steps):
                jitter[i] = decay * jitter[i - 1] + drive * rng.normal()
        else:
            # Pure random walk (no guiding correction)
            jitter = np.cumsum(rng.normal(0.0, jitter_sigma * np.sqrt(dt), n_steps))
        jx = jitter * cos_a
        jy = jitter * (-sin_a)
    else:
        jx = jy = np.zeros(n_steps)

    y_grid = np.arange(ny, dtype=float)
    x_grid = np.arange(nx, dtype=float)
    image = np.zeros((ny, nx))
    four_sigma = 4.0 * sigma

    for i, t in enumerate(t_vals):
        cx = x0 + t * sin_a + jx[i]
        cy = y0 + t * cos_a + jy[i]
        # Restrict evaluation to within 4 sigma of the PSF centre
        ylo = max(0, int(cy - four_sigma))
        yhi = min(ny, int(cy + four_sigma) + 2)
        xlo = max(0, int(cx - four_sigma))
        xhi = min(nx, int(cx + four_sigma) + 2)
        if ylo >= yhi or xlo >= xhi:
            continue
        dy = y_grid[ylo:yhi, np.newaxis] - cy
        dx = x_grid[np.newaxis, xlo:xhi] - cx
        image[ylo:yhi, xlo:xhi] += np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))

    total = image.sum()
    if total > 0:
        image *= flux / total
    return image
