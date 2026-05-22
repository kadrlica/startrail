# startrail

Software for simulating and fitting stars observed with continuous (open-shutter) CCD readout.
During continuous readout, stars appear as elongated trails rather than point sources;
this package models and fits those trails to extract astrometric and photometric measurements.

## Overview

- **Simulate** — generate realistic synthetic CCD images with open-shutter star trails,
  including Gaussian and Moffat PSF models, Poisson photon noise, sky background, read noise,
  and telescope tracking jitter (Ornstein-Uhlenbeck process)
- **Fit** — detect and fit trails in images (real or simulated) using a parametric forward
  model optimised with `scipy.optimize`
- **Validate** — Jupyter notebooks serve as integration tests and visual verification

## Installation

```bash
# Create and activate environment
mamba create -n startrail python=3.11
mamba activate startrail

# photutils must be installed via conda-forge (pip wheel build fails)
conda install -c conda-forge photutils

# Install package with development dependencies
pip install -e ".[dev]"
```

## Usage

```python
from astropy.table import Table
from startrail.simulate import make_image

# Define a single star
star = Table({'x': [50.0], 'y': [500.0], 'angle': [0.0],
              'length': [900.0], 'flux': [1e6]})

# Simulate a 1000-row CCD image with Poisson noise, sky, and read noise
image = make_image(
    (1000, 100), star, fwhm=3.0,
    sky_level=200.0, read_noise=8.0, bias=1000.0,
    jitter_sigma=0.5, tau=50.0, seed=42,
)
```

```python
from startrail.detect import detect_sources
from startrail.fit import fit_image

sources = detect_sources(image, threshold_sigma=5.0, fwhm_guess=3.0)
catalog = fit_image(image, sources)
print(catalog['x0', 'y0', 'angle', 'length', 'flux'])
```

## Package structure

```
startrail/
├── simulate/
│   ├── psf.py       # Gaussian and Moffat PSF models
│   ├── trail.py     # Trail generation (PSF × line kernel + OU jitter)
│   ├── noise.py     # Poisson, sky, read noise, bias
│   └── image.py     # Multi-star image assembly
├── models.py        # TrailModel dataclass (forward model)
├── detect.py        # Source detection (photutils wrapper)
├── fit.py           # Trail fitting → astropy Table
├── io.py            # FITS and catalog I/O
└── utils.py         # Coordinate helpers
```

## Development

```bash
# Run tests
pytest

# Execute all notebooks and convert to HTML
for nb in notebooks/*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
    jupyter nbconvert --to html "$nb"
done
```

## Key concepts

**Open-shutter readout** — the CCD reads out continuously while the shutter is open,
causing stars to trail along the readout direction as charge is transferred row by row.

**Trail model** — a star trail is the convolution of the stellar PSF with a line kernel
whose length is proportional to the readout time.

**Jitter** — telescope tracking is imperfect. The star's position follows an
Ornstein-Uhlenbeck process: a mean-reverting random walk where the guider pulls the
star back toward the nominal pointing on a characteristic timescale τ. Jitter is
modelled as isotropic (azimuthally symmetric) in x and y.
