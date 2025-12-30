"""Model zoo (registered via sciml_pipeline.registry)."""

from .mlp import MLP
from .fourier_mlp import FourierFeatureMLP
from .cnn1d import CNN1D
from .param_vector import ParamVector

# Side-effect: import modules to register factories
from . import mlp as _mlp  # noqa: F401
from . import fourier_mlp as _fourier_mlp  # noqa: F401
from . import cnn1d as _cnn1d  # noqa: F401
from . import param_vector as _param_vector  # noqa: F401

__all__ = ["MLP", "FourierFeatureMLP", "CNN1D", "ParamVector"]

