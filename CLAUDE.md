# startrail

Software for fitting stars observed with continuous (open-shutter) CCD readout.
During continuous readout, stars appear as elongated trails rather than point sources;
this package models and fits those trails to extract astrometric and photometric measurements.

## Tech stack

- Python (conda/mamba environment)
- astropy — FITS I/O, WCS, sky coordinates, units
- numpy / scipy — array math, optimization, curve fitting
- matplotlib — visualization and diagnostic plots
- photutils — source detection and aperture/PSF photometry

## Development

```bash
# Create environment
mamba create -n startrail python=3.11
mamba activate startrail
pip install -e ".[dev]"

# Run tests
pytest

# Run a single test file
pytest tests/test_foo.py -v
```

## Notebooks

Jupyter notebooks live in `notebooks/`. They serve as integration tests and visual validation:
each notebook is executed top-to-bottom and converted to HTML whenever it is updated.
The resulting `.html` files appear alongside the notebooks and are gitignored.

```bash
# Manually run and render all notebooks
for nb in notebooks/*.ipynb; do
    jupyter nbconvert --to html --execute "$nb"
done
```

Notebooks should be self-contained and runnable from the repo root.
Use relative paths or small bundled sample FITS files for input data.

## Project conventions

- FITS images are the primary data format; use `astropy.io.fits` for I/O
- Physical units should use `astropy.units`; sky coordinates use `astropy.coordinates`
- Fit results and catalogs are stored as `astropy.table.Table` (FITS or ECSV output)
- Plots go in `plots/`; intermediate data products go in `output/`
- Keep fitting logic separate from I/O; fitting functions should operate on numpy arrays
- The project should define a simulator intended to make realistic simulated images
- The fitting should operate on both real and simulated images

## Key concepts

- **Open-shutter / continuous readout**: the CCD reads out continuously while the shutter
  is open, causing stars to trail along the readout direction as charge is transferred
- **Trail-length**: The star trails will generally be very long (>1000 rows).
- **Trail model**: a star trail is a convolution of the stellar PSF with a line kernel
  of length proportional to the star's flux and the readout speed
- **Fitting**: trails are fit in pixel space
- **Jitter**: The telescope tracking is imperfect and the star will move
