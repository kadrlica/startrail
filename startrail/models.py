from dataclasses import dataclass

import numpy as np


@dataclass
class TrailModel:
    """Parametric model for a single star trail.

    The trail is the integral of a Gaussian PSF along a straight-line path
    (convolution of the PSF with a line-segment kernel).

    Attributes
    ----------
    x0, y0 : float
        Trail centre in pixel coordinates.
    angle : float
        Trail angle in radians (0 = along y / readout axis).
    length : float
        Trail length in pixels.
    fwhm : float
        PSF FWHM in pixels.
    flux : float
        Total trail flux in counts.
    """

    x0: float
    y0: float
    angle: float
    length: float
    fwhm: float
    flux: float

    def evaluate(self, shape):
        """Return the model evaluated on a pixel grid.

        Parameters
        ----------
        shape : (ny, nx)

        Returns
        -------
        numpy.ndarray
        """
        from .simulate.trail import make_trail

        return make_trail(
            shape,
            self.x0,
            self.y0,
            self.angle,
            self.length,
            self.fwhm,
            self.flux,
        )

    def residuals(self, image):
        """Return ``image - model`` residuals.

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return image - self.evaluate(image.shape)

    def as_dict(self):
        """Return parameters as a plain dict (for building astropy Table rows)."""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "angle": self.angle,
            "length": self.length,
            "fwhm": self.fwhm,
            "flux": self.flux,
        }
