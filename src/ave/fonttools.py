"""Classes related to the FontTools library."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from numpy.typing import NDArray

from ave.consts import AvGlyphCmds


class FontHelper:
    """
    Class to provide various static methods related to font handling.
    """

    @staticmethod
    def get_default_axes_values(variable_font: TTFont) -> Dict[str, float]:
        """
        Get the default axes values of a given variable TTFont.
        Use returned values to instantiate a new font.
        """
        default_axes_values: Dict[str, float] = {}
        if variable_font.get("fvar") is not None:
            for axis in variable_font["fvar"].axes:  # type: ignore
                default_axes_values[axis.axisTag] = axis.defaultValue
        else:
            raise ValueError("Variable font has no 'fvar' table.")
        return default_axes_values

    @staticmethod
    def instantiate_ttfont(variable_font: TTFont, axes_values: Dict[str, float]) -> TTFont:
        """
        Instantiate a new font from a given variable TTFont and the given axes_values.
        Returns a new TTFont.
        Example for axes_values: {"wght": 700, "wdth": 25, "GRAD": 100}

        Args:
            variable_font (TTFont): The variable font to instantiate.
            axes_values (Dict[str, float]): A dictionary mapping axis names to values.

        Returns:
            TTFont: The instantiated font.
        """
        instantiate_axes_values = FontHelper.get_default_axes_values(variable_font)
        instantiate_axes_values.update(axes_values)
        return instancer.instantiateVariableFont(variable_font, instantiate_axes_values)


###############################################################################
# Pens
###############################################################################
class AvPolylinePen(BasePen):
    """
    This pen is used to convert curves to line segments.
    It records the curve as a sequence of line segments.
    """

    def __init__(self, glyphSet, steps=10):
        """
        Initializes a new instance of the class.

        Args:
            glyphSet (GlyphSet): The glyph set to use.
            steps (int, optional): The number of steps to use for polygonization. Defaults to 10.
        """
        super().__init__(glyphSet)
        self.steps = steps
        self.recording_pen = RecordingPen()

    def _moveTo(self, pt):
        self.recording_pen.moveTo(pt)

    def _lineTo(self, pt):
        self.recording_pen.lineTo(pt)

    def _curveToOne(self, pt1, pt2, pt3):
        self._polygonize_cubic_bezier([self._getCurrentPoint(), pt1, pt2, pt3])

    def _qCurveToOne(self, pt1, pt2):
        self._polygonize_quadratic_bezier([self._getCurrentPoint(), pt1, pt2])

    def _closePath(self):
        self.recording_pen.closePath()

    def _endPath(self):
        self.recording_pen.endPath()

    def _polygonize_quadratic_bezier(self, points):
        pt0, pt1, pt2 = points
        for t in [i / self.steps for i in range(1, self.steps + 1)]:
            x = (1 - t) ** 2 * pt0[0] + 2 * (1 - t) * t * pt1[0] + t**2 * pt2[0]
            y = (1 - t) ** 2 * pt0[1] + 2 * (1 - t) * t * pt1[1] + t**2 * pt2[1]
            self._lineTo((x, y))

    def _polygonize_cubic_bezier(self, points):
        pt0, pt1, pt2, pt3 = points
        for t in [i / self.steps for i in range(1, self.steps + 1)]:
            x = (1 - t) ** 3 * pt0[0] + 3 * (1 - t) ** 2 * t * pt1[0] + 3 * (1 - t) * t**2 * pt2[0] + t**3 * pt3[0]
            y = (1 - t) ** 3 * pt0[1] + 3 * (1 - t) ** 2 * t * pt1[1] + 3 * (1 - t) * t**2 * pt2[1] + t**3 * pt3[1]
            self._lineTo((x, y))


class AvGlyphPtsCmdsPen(BasePen):
    """
    Pen that records glyph drawing commands and their points in a compact
    representation: a flat list of points and a parallel list of SVG command
    letters. The points are recorded in the order the
    commands supply them; consumers must know how many coordinates each
    command consumes (e.g. 'C' consumes 3 points = 6 numbers).
    Supports the commands: M, L, C, Q, Z (all absolute).

    After drawing a glyph with this pen, use the `.points` and `.commands`
    properties to access the results.
    """

    def __init__(self, glyphSet):
        super().__init__(glyphSet)
        # numpy array of shape (n, 2) storing (x,y) points in the order they were emitted
        self._points: NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
        # list of command letters (M, L, C, Q, Z)
        self._commands: List[AvGlyphCmds] = []

    # BasePen callback methods -------------------------------------------------
    def _moveTo(self, pt: Tuple[float, float]):
        self._commands.append("M")
        self._points = np.vstack([self._points, [[float(pt[0]), float(pt[1])]]])

    def _lineTo(self, pt: Tuple[float, float]):
        self._commands.append("L")
        self._points = np.vstack([self._points, [[float(pt[0]), float(pt[1])]]])

    def _curveToOne(self, pt1: Tuple[float, float], pt2: Tuple[float, float], pt3: Tuple[float, float]):
        # cubic bezier: two control points and an end point
        self._commands.append("C")
        self._points = np.vstack(
            [
                self._points,
                [[float(pt1[0]), float(pt1[1])]],
                [[float(pt2[0]), float(pt2[1])]],
                [[float(pt3[0]), float(pt3[1])]],
            ]
        )

    def _qCurveToOne(self, pt1: Tuple[float, float], pt2: Tuple[float, float]):
        # quadratic bezier: one control point and an end point
        self._commands.append("Q")
        self._points = np.vstack([self._points, [[float(pt1[0]), float(pt1[1])]], [[float(pt2[0]), float(pt2[1])]]])

    def _closePath(self):
        self._commands.append("Z")

    def _endPath(self):
        # treat endPath like closePath for command recording (no coordinates)
        self._commands.append("Z")

    # Public accessors ---------------------------------------------------------
    @property
    def commands(self) -> List[AvGlyphCmds]:
        """Return the recorded commands as a list (uppercase commands)."""
        return self._commands

    @property
    def points(self) -> NDArray[np.float64]:
        """Return recorded points as an (n_points, 2) ndarray of float64."""
        return self._points

    def reset(self) -> None:
        """Clear recorded commands and points."""
        self._points = np.empty((0, 2), dtype=np.float64)
        self._commands = []

    def summary(self) -> str:
        """Small debug helper summarizing the recorded data."""
        return f"commands={self._commands}, points.shape={self.points.shape}"
