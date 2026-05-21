from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


def read_image(path):
    """Read a FITS image from disk.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    data : numpy.ndarray (float64)
    header : astropy.io.fits.Header
    """
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()
    return data, header


def write_image(path, data, header=None):
    """Write a numpy array to a FITS file.

    Parameters
    ----------
    path : str or Path
    data : numpy.ndarray
    header : astropy.io.fits.Header or None
    """
    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
    hdu.writeto(path, overwrite=True)


def read_catalog(path):
    """Read a trail catalog from a FITS or ECSV file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    astropy.table.Table
    """
    return Table.read(path)


def write_catalog(catalog, path):
    """Write a trail catalog to disk.

    The output format is determined by the file extension:
    ``.ecsv`` → ASCII ECSV, anything else → FITS binary table.

    Parameters
    ----------
    catalog : astropy.table.Table
    path : str or Path
    """
    path = Path(path)
    if path.suffix == ".ecsv":
        catalog.write(path, format="ascii.ecsv", overwrite=True)
    else:
        catalog.write(path, format="fits", overwrite=True)
