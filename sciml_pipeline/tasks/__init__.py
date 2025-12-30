"""Task zoo (registered via sciml_pipeline.registry)."""

# Side-effect: import modules to register task builders
from . import classification_npz as _classification_npz  # noqa: F401
from . import pinn_poisson1d as _pinn_poisson1d  # noqa: F401
from . import inverse_deconvolution1d as _inverse_deconvolution1d  # noqa: F401

