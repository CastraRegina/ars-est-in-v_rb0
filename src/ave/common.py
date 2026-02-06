"""Central module containing constants and definitions for SVG processing and geometry."""

from __future__ import annotations

import os
import sys
import warnings
from enum import Enum, auto
from functools import wraps
from typing import List, Literal

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

# Standard affine transformation: [a00, a01, a10, a11, b0, b1]
# Represents matrix:
#   | a00 a01 b0 |
#   | a10 a11 b1 |
#   |   0   0  1 |
AffineTransform = List[float]


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


def deprecated(reason: str = ""):
    """Declare a function as deprecated and emit a warning on each call."""

    def decorator(func):
        message = f"{func.__name__} is deprecated. {reason}".strip()

        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def sgn_sci(value: float, precision: int = 3, always_positive: bool = False) -> str:
    """Return sign-aligned scientific-notation text with zero-padded exponent.
        Values are rounded using Python’s round-half-to-even rule (banker’s rounding),
        so halfway cases like 1234.5 stay at 1.234 rather than 1.235.

    Examples:
        sgn_sci(0.1234) =  1.234e-001
        sgn_sci(-12.34) = -1.234e+001
        sgn_sci(1234.5) =  1.234e+003
        sgn_sci(1234.6) =  1.235e+003
        sgn_sci(2345.5) =  2.346e+003
        sgn_sci(3456.5) =  3.456e+003

    Args:
        value: Number to format.
        precision: Digits to keep to the right of the decimal. Defaults to 3.
        always_positive: Remove the placeholder for a leading sign when True.
            Useful for compact representations of values that are guaranteed to
            be non-negative. Defaults to False.

    Returns:
        The formatted string with leading sign placeholder and fixed exponent width.
    """
    mantissa, exponent = f"{abs(value):.{precision}e}".split("e")
    if value < 0:
        sign_char = "-"
    elif always_positive:
        sign_char = ""
    else:
        sign_char = " "
    exponent_sign = exponent[0]
    exponent_value = exponent[1:].rjust(3, "0")
    return f"{sign_char}{mantissa}e{exponent_sign}{exponent_value}"


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
    print("sgn_sci(0.1234) =", sgn_sci(0.1234))
    print("sgn_sci(-12.34) =", sgn_sci(-12.34))
    print("sgn_sci(1234.5) =", sgn_sci(1234.5))
    print("sgn_sci(1234.6) =", sgn_sci(1234.6))
    print("sgn_sci(2345.5) =", sgn_sci(2345.5))
    print("sgn_sci(3456.5) =", sgn_sci(3456.5))

    print()


if __name__ == "__main__":
    main()
