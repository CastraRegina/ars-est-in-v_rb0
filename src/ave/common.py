"""Central module containing constants and definitions for SVG processing and geometry."""

from __future__ import annotations

import os
import sys
from enum import Enum, auto
from typing import Literal

###############################################################################
# Types
###############################################################################


AvGlyphCmds = Literal[  # Type-Definition for SvgPath-Commands used in AvGlyph
    # MoveTo (2) - start a new subpath and move the current point to (x,y)
    "M",
    # LineTo (2) - draw a straight line from the current point to (x,y)
    "L",
    # Cubic Bezier To (6) - draw a cubic Bezier curve with two control points and an endpoint (x,y)
    "C",
    # Quadratic Bezier To (4) - draw a quadratic Bezier curve with one control point and an endpoint (x,y)
    "Q",
    # ClosePath (0) - close subpath by drawing a line from the current point to start point
    "Z",
]


###############################################################################
# Enums and Consts
###############################################################################


class Align(Enum):
    """Enum to define text alignment options."""

    LEFT = auto()
    RIGHT = auto()
    BOTH = auto()


###############################################################################
# Functions
###############################################################################


def main() -> None:
    """Display system information and alignment enum values.

    This function prints the Python path, PYTHONPATH environment variable,
    and demonstrates the Align enum values.
    """
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
