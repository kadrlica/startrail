import numpy as np


def cutout(image, x0, y0, half_size):
    """Extract a square cutout from an image.

    Returns
    -------
    cutout : numpy.ndarray
    slices : (row_slice, col_slice) into the original image
    """
    ny, nx = image.shape
    y0i, x0i = int(round(y0)), int(round(x0))
    r0 = max(0, y0i - half_size)
    r1 = min(ny, y0i + half_size + 1)
    c0 = max(0, x0i - half_size)
    c1 = min(nx, x0i + half_size + 1)
    return image[r0:r1, c0:c1], (slice(r0, r1), slice(c0, c1))


def bbox_contains(shape, x, y):
    """Return True if pixel (x, y) is within the image bounds."""
    ny, nx = shape
    return (0 <= x < nx) and (0 <= y < ny)


def fwhm_to_sigma(fwhm):
    """Convert FWHM to Gaussian sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
