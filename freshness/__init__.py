"""Local produce freshness classifier."""
import os
# Albumentations must not contact an update server during offline use.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
__version__ = "1.0.0"
