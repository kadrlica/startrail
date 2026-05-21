# Sample data

Small FITS files used by notebooks and tests.

Files placed here should be minimal in size (< 1 MB each) and represent
realistic open-shutter CCD frames or extracted trail cutouts.

Notebooks and tests that need synthetic data should generate it inline
using `startrail.simulate.make_image` rather than committing large FITS
files to the repository.
