__all__ = [
    "ROOT_DIR"
]

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)