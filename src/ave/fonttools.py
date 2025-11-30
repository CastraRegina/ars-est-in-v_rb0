"""Classes related to the FontTools library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union, cast

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from numpy.typing import NDArray

from ave.common import AvGlyphCmds
from ave.geom import BezierCurve

# from fontTools.pens.recordingPen import RecordingPen


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
# class AvPolylinePen(BasePen):
#     """
#     This pen is used to convert curves to line segments.
#     It records the curve as a sequence of line segments.
#     """

#     def __init__(self, glyphSet, steps=10):
#         """
#         Initializes a new instance of the class.

#         Args:
#             glyphSet (GlyphSet): The glyph set to use.
#             steps (int, optional): The number of steps to use for polygonization. Defaults to 10.
#         """
#         super().__init__(glyphSet)
#         self.steps = steps
#         self.recording_pen = RecordingPen()

#     def _moveTo(self, pt: Tuple[float, float]):
#         self.recording_pen.moveTo(pt)

#     def _lineTo(self, pt: Tuple[float, float]):
#         self.recording_pen.lineTo(pt)

#     def _getCurrentPoint(self) -> Tuple[float, float]:
#         pt = super()._getCurrentPoint()
#         # if the point is a tuple of two floats (or ints), return it
#         if isinstance(pt, tuple) and len(pt) == 2 and all(isinstance(x, (int, float, np.float64)) for x in pt):
#             return pt
#         raise ValueError(f"Invalid point {pt} in _getCurrentPoint")

#     def _curveToOne(self, pt1: Tuple[float, float], pt2: Tuple[float, float], pt3: Tuple[float, float]):
#         self._polygonize_cubic_bezier([self._getCurrentPoint(), pt1, pt2, pt3])

#     def _qCurveToOne(self, pt1: Tuple[float, float], pt2: Tuple[float, float]):
#         self._polygonize_quadratic_bezier([self._getCurrentPoint(), pt1, pt2])

#     def _closePath(self):
#         self.recording_pen.closePath()

#     def _endPath(self):
#         self.recording_pen.endPath()

#     def _polygonize_quadratic_bezier(self, points: Sequence[Tuple[float, float]]):
#         pt0, pt1, pt2 = points
#         for t in [i / self.steps for i in range(1, self.steps + 1)]:
#             x = (1 - t) ** 2 * pt0[0] + 2 * (1 - t) * t * pt1[0] + t**2 * pt2[0]
#             y = (1 - t) ** 2 * pt0[1] + 2 * (1 - t) * t * pt1[1] + t**2 * pt2[1]
#             self._lineTo((x, y))

#     def _polygonize_cubic_bezier(self, points: Sequence[Tuple[float, float]]):
#         pt0, pt1, pt2, pt3 = points
#         for t in [i / self.steps for i in range(1, self.steps + 1)]:
#             x = (1 - t) ** 3 * pt0[0] + 3 * (1 - t) ** 2 * t * pt1[0] + 3 * (1 - t) * t**2 * pt2[0] + t**3 * pt3[0]
#             y = (1 - t) ** 3 * pt0[1] + 3 * (1 - t) ** 2 * t * pt1[1] + 3 * (1 - t) * t**2 * pt2[1] + t**3 * pt3[1]
#             self._lineTo((x, y))


@dataclass
class AvGlyphPtsCmdsPen(BasePen):
    """
    Records glyph drawing commands and their points in a compact representation.
    The points are recorded in the order the commands supply them.
    Input points are supposed to be Tuple[float, float]
    while the result points are NDArray[np.float64] of shape (n, 3).

    Supports the commands: M, L, C, Q, Z (all absolute).
    Points ".points" dimension is 3: (x, y, type).
    Type is 0.0 for start/end point, 2.0 for quadratic, 3.0 for cubic curve point.

    Access the results via `.points` and `.commands` after drawing a glyph with this pen.
    """

    _points: NDArray[np.float64]  # numpy array of shape (n, 3) = (x, y, type)
    _commands: List[AvGlyphCmds]
    _polygonize_steps: int

    def __init__(self, glyphSet, polygonize_steps: int = 0):
        """
        Initialize the AvGlyphPtsCmdsPen.

        Parameters:
            glyphSet (GlyphSet): The glyph set to use.
            polygonize_steps (int, optional): The number of steps to use for polygonization.
                Defaults to 0 = no polygonization. Steps = number of segments (lines) the curve will be divided into.

        Notes:
            If polygonize_steps is 0 the commands could contain also curves
            If polygonize_steps is greater than 0, then curves will be polygonized
        """
        super().__init__(glyphSet)
        self._polygonize_steps = polygonize_steps
        # numpy array of shape (n, 3) storing (x,y,type) points in the order they were emitted
        self._points: NDArray[np.float64] = np.empty((0, 3), dtype=np.float64)
        # list of command letters (M, L, C, Q, Z)
        self._commands: List[AvGlyphCmds] = []

    # BasePen callback methods -------------------------------------------------
    def _moveTo(self, pt: Tuple[float, float]):
        self._commands.append("M")
        self._points = np.vstack(
            [
                self._points,
                [np.float64(pt[0]), np.float64(pt[1]), np.float64(0.0)],
            ],
        )

    def _line_to_with_type(self, pt: Tuple[float, float], point_type: float):
        self._commands.append("L")
        self._points = np.vstack(
            [
                self._points,
                [np.float64(pt[0]), np.float64(pt[1]), np.float64(point_type)],
            ],
        )

    def _lineTo(self, pt: Tuple[float, float]):
        self._line_to_with_type(pt, 0.0)

    def _qCurveToOne(self, pt1: Tuple[float, float], pt2: Tuple[float, float]):
        # quadratic bezier: one control point and an end point
        if self._polygonize_steps > 0:
            self._polygonize_quadratic_bezier([self._getCurrentPoint(), pt1, pt2])
        else:
            self._commands.append("Q")
            self._points = np.vstack(
                [
                    self._points,
                    [np.float64(pt1[0]), np.float64(pt1[1]), np.float64(2.0)],
                    [np.float64(pt2[0]), np.float64(pt2[1]), np.float64(0.0)],
                ]
            )

    def _curveToOne(self, pt1: Tuple[float, float], pt2: Tuple[float, float], pt3: Tuple[float, float]):
        # cubic bezier: two control points and an end point
        if self._polygonize_steps > 0:
            self._polygonize_cubic_bezier([self._getCurrentPoint(), pt1, pt2, pt3])
        else:
            self._commands.append("C")
            self._points = np.vstack(
                [
                    self._points,
                    [np.float64(pt1[0]), np.float64(pt1[1]), np.float64(3.0)],
                    [np.float64(pt2[0]), np.float64(pt2[1]), np.float64(3.0)],
                    [np.float64(pt3[0]), np.float64(pt3[1]), np.float64(0.0)],
                ]
            )

    def _closePath(self):
        self._commands.append("Z")

    def _endPath(self):
        # treat endPath like closePath for command recording (no coordinates)
        self._commands.append("Z")

    def _getCurrentPoint(self) -> Tuple[float, float]:
        pt = super()._getCurrentPoint()
        # if the point is a tuple of two floats (or ints), return it
        if isinstance(pt, tuple) and len(pt) == 2 and all(isinstance(x, (int, float, np.float64)) for x in pt):
            return pt
        raise ValueError(f"Invalid point {pt} in _getCurrentPoint")

    def _polygonize_quadratic_bezier(self, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]]):
        new_points = BezierCurve.polygonize_quadratic_bezier(points, self._polygonize_steps)
        self._points = np.vstack([self._points, new_points])
        self._commands.extend(cast(List[AvGlyphCmds], ["L"] * self._polygonize_steps))

        # pt0, pt1, pt2 = points
        # steps = self._polygonize_steps

        # if steps <= 150:
        #     # Ultra-fast pure Python path
        #     inv_steps = 1.0 / steps
        #     for i in range(1, steps + 1):
        #         t = i * inv_steps
        #         omt = 1.0 - t
        #         omt2 = omt * omt
        #         t2 = t * t
        #         x = omt2 * pt0[0] + 2 * omt * t * pt1[0] + t2 * pt2[0]
        #         y = omt2 * pt0[1] + 2 * omt * t * pt1[1] + t2 * pt2[1]
        #         self._line_to_with_type((x, y), 2.0)
        # else:
        #     # fast NumPy path
        #     t = np.arange(1, steps + 1) / steps
        #     omt = 1 - t
        #     x = omt**2 * pt0[0] + 2 * omt * t * pt1[0] + t**2 * pt2[0]
        #     y = omt**2 * pt0[1] + 2 * omt * t * pt1[1] + t**2 * pt2[1]
        #     new_points = np.column_stack([x, y, np.full(steps, 2.0, dtype=np.float64)])
        #     self._points = np.vstack([self._points, new_points])
        #     self._commands.extend(cast(List[AvGlyphCmds], ["L"] * steps))

    def _polygonize_cubic_bezier(self, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]]):
        new_points = BezierCurve.polygonize_cubic_bezier(points, self._polygonize_steps)
        self._points = np.vstack([self._points, new_points])
        self._commands.extend(cast(List[AvGlyphCmds], ["L"] * self._polygonize_steps))

        # pt0, pt1, pt2, pt3 = points
        # steps = self._polygonize_steps

        # if steps <= 75:
        #     # Ultra-fast pure Python path
        #     inv_steps = 1.0 / steps
        #     for i in range(1, steps + 1):
        #         t = i * inv_steps
        #         omt = 1.0 - t
        #         omt2 = omt * omt
        #         omt3 = omt2 * omt
        #         t2 = t * t
        #         t3 = t2 * t
        #         x = omt3 * pt0[0] + 3 * omt2 * t * pt1[0] + 3 * omt * t2 * pt2[0] + t3 * pt3[0]
        #         y = omt3 * pt0[1] + 3 * omt2 * t * pt1[1] + 3 * omt * t2 * pt2[1] + t3 * pt3[1]
        #         self._line_to_with_type((x, y), 3.0)
        # else:
        #     # fast NumPy path
        #     t = np.arange(1, steps + 1) / steps  # t from 1/N to 1
        #     omt = 1.0 - t
        #     x = omt**3 * pt0[0] + 3 * omt**2 * t * pt1[0] + 3 * omt * t**2 * pt2[0] + t**3 * pt3[0]
        #     y = omt**3 * pt0[1] + 3 * omt**2 * t * pt1[1] + 3 * omt * t**2 * pt2[1] + t**3 * pt3[1]
        #     new_points = np.column_stack([x, y, np.full(steps, 3.0, dtype=np.float64)])
        #     self._points = np.vstack([self._points, new_points])
        #     self._commands.extend(cast(List[AvGlyphCmds], ["L"] * steps))

    @property
    def commands(self) -> List[AvGlyphCmds]:
        """Return the recorded commands as a list (uppercase commands)."""
        return self._commands

    @property
    def points(self) -> NDArray[np.float64]:
        """Return recorded points as an (n_points, 3) ndarray of float64."""
        return self._points

    def reset(self) -> None:
        """Clear recorded commands and points."""
        self._points = np.empty((0, 3), dtype=np.float64)
        self._commands = []
