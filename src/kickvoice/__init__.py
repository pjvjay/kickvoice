"""kickvoice — Speaker identification on Kick.com live streams."""

from kickvoice._openmp import allow_duplicate_openmp

__version__ = "0.2.0"

# Intel Macs only; has to happen before torch or ctranslate2 is imported
# anywhere in the process (see kickvoice/_openmp.py).
allow_duplicate_openmp()
