"""Python bindings for SZo error-bounded lossy compression."""

from pyszo.sz import sz
from pyszo.pyConfig import szoConfig
from pyszo.pyConfigEnums import szoErrorBoundMode, szoAlgorithm, szoInterpAlgorithm

__all__ = ["sz", "szoConfig", "szoErrorBoundMode", "szoAlgorithm", "szoInterpAlgorithm"]
