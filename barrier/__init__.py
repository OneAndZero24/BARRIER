"""
barrier — shared core of the BARRIER project.

The core algorithm ("InTAct", Interval-based Task Activation Consolidation)
lives in :mod:`barrier.intact` as the :class:`UnlearnIntervalProtection`
class.  Every experiment directory (Classification/, DDPM/, SD/, Flux/)
imports it from here; no experiment re-implements the protection loss.

Importing this package also runs :mod:`barrier.cache` (a no-op side effect
that redirects model/data caches off the home filesystem), so it is safe to
import ``barrier`` before any torch / transformers / diffusers import.
"""

from . import cache  # noqa: F401  — must run before any model library import
from .intact import UnlearnIntervalProtection, classification_forward_fn

__all__ = ["UnlearnIntervalProtection", "classification_forward_fn"]