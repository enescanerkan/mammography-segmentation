"""Mammography segmentation, landmarks and positioning assessment.

Every subpackage imports its heavy dependencies lazily, so light users (``breast_seg.geometry``) do not pull in
torch / ultralytics.
"""
__version__ = "1.0.0"

__all__ = ["__version__"]
