"""MemoryDNA: small-RNA epigenetic inheritance + evolutionary computation."""

from .ga import GAConfig, ModelKind, run_population
from .model import SilvaParameters

__all__ = ["GAConfig", "ModelKind", "SilvaParameters", "run_population"]
