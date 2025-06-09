#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED: This entry point is deprecated.

Please use 'python run.py realtime' instead.

This file is kept for backward compatibility but will be removed in future versions.
"""

import warnings
import sys
from pathlib import Path

# Show deprecation warning
warnings.warn(
    "DEPRECATED: main.py is deprecated. Please use 'python run.py realtime' instead. "
    "This file will be removed in future versions.",
    DeprecationWarning,
    stacklevel=1
)

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the real application
if __name__ == "__main__":
    from src.apps.main import main as app_main
    app_main()
