"""Python bindings for SZo error-bounded lossy compression."""

from pyszo.szo import szo
from pyszo.pyConfig import szoConfig
from pyszo.pyConfigEnums import szoErrorBoundMode, szoAlgorithm, szoInterpAlgorithm

__all__ = ["szo", "szoConfig", "szoErrorBoundMode", "szoAlgorithm", "szoInterpAlgorithm"]
