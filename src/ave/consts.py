"""Central module containing constants and definitions"""

from __future__ import annotations

import os
import sys
from enum import Enum, auto


class Align(Enum):
    """
    Enum to define alignments
    """

    LEFT = auto()
    RIGHT = auto()
    BOTH = auto()


def main():
    """Main"""
    print("sys.path:  ", sys.path)
    print()
    print("PYTHONPATH:", os.environ["PYTHONPATH"])
    print()
    print()

    print(Align.LEFT, Align.LEFT.value)
    print(Align.RIGHT, Align.RIGHT.value)
    print(Align.BOTH, Align.BOTH.value)

    print()


if __name__ == "__main__":
    main()
