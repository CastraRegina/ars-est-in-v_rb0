"""Central module for consts and definitions"""

# from __future__ import annotations

import os
import sys
from enum import Enum, auto

# if __name__ == "__main__":
#     sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class Polygonize(Enum):
    """Enum to define types of polygonization"""

    BY_ANGLE = auto()
    UNIFORM = auto()


class Align(Enum):
    """Enum to define alignments"""

    LEFT = auto()
    RIGHT = auto()
    BOTH = auto()


POLYGONIZE_UNIFORM_NUM_POINTS = 10  # minimum 2 = (start, end)
POLYGONIZE_ANGLE_MAX_DEG = 5  # 2 # difference of two derivatives less than
POLYGONIZE_ANGLE_MAX_STEPS = 9  # 9
POLYGONIZE_TYPE = Polygonize.BY_ANGLE


def main():
    """Main"""
    print("sys.path:  ", sys.path)
    print()
    print("PYTHONPATH:", os.environ["PYTHONPATH"])
    print()
    print()

    print(Polygonize.BY_ANGLE, Polygonize.BY_ANGLE.value)
    print(Polygonize.UNIFORM, Polygonize.UNIFORM.value)

    print(Align.LEFT, Align.LEFT.value)
    print(Align.RIGHT, Align.RIGHT.value)
    print(Align.BOTH, Align.BOTH.value)

    print("POLYGONIZE_UNIFORM_NUM_POINTS", POLYGONIZE_UNIFORM_NUM_POINTS)
    print("POLYGONIZE_ANGLE_MAX_DEG", POLYGONIZE_ANGLE_MAX_DEG)
    print("POLYGONIZE_ANGLE_MAX_STEPS", POLYGONIZE_ANGLE_MAX_STEPS)
    print("POLYGONIZE_TYPE", POLYGONIZE_TYPE)


if __name__ == "__main__":
    main()
