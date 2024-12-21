"""Classes related to the FontTools library."""

from __future__ import annotations

from typing import Dict

from fontTools.pens.basePen import BasePen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer


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


# =============================================================================
# Pens
# =============================================================================
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

    # def draw(self, pen):
    #     for command in self.recordingPen.value:
    #         pen._callCommand(command)

    # TODO: for SVG-commands use fontTools.pens.svgPathPen.SVGPathPen
    # def svg_commands(self):
    #     svg_commands = []
    #     for command, points in self.recording_pen.value:
    #         if command == "moveTo":
    #             svg_commands.append(f"M {points[0][0]} {points[0][1]}")
    #         elif command == "lineTo":
    #             svg_commands.append(f"L {points[0][0]} {points[0][1]}")
    #         elif command == "curveTo":
    #             svg_commands.append(
    #                 f"C {points[0][0][0]} {points[0][0][1]} {points[1][0]} {points[1][1]} {points[2][0]} {points[2][1]}"
    #             )
    #         elif command == "qCurveTo":
    #             svg_commands.append(f"Q {points[0][0]} {points[0][1]} {points[1][0]} {points[1][1]}")
    #         elif command == "closePath":
    #             svg_commands.append("Z")
    #     return " ".join(svg_commands)
