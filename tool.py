"""Backward-compatible launcher for the Camera Viewer project."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from camera_viewer.app import main


if __name__ == "__main__":
    main()
