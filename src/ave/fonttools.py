"""Font processing utilities and FontTools library integration for OpenType fonts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union, cast

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from numpy.typing import NDArray

from ave.bezier import BezierCurve
from ave.common import AvGlyphCmds

# from fontTools.pens.recordingPen import RecordingPen


class FontHelper:
    """Class to provide various static methods related to font handling.

    Provides utilities for font loading, glyph extraction, and font analysis
    using the FontTools library.
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

    @staticmethod
    def get_kerning_value(ttfont: TTFont, left_char: str, right_char: str) -> int:
        """Get kerning value for a character pair from the GPOS table.

        This function extracts kerning information from OpenType fonts. For variable fonts,
        the kerning value depends on the current instance/axis values of the font.

        Args:
            ttfont (TTFont): A fontTools TTFont object. For variable fonts, this should
                be an instantiated font with specific axis values already applied.
            left_char (str): The left character in the pair (e.g., 'A' in 'AV').
            right_char (str): The right character in the pair (e.g., 'V' in 'AV').

        Returns:
            int: Kerning value in font units.
                - Negative values: characters move closer together
                - Positive values: characters move farther apart
                - Zero: no kerning adjustment

        Notes:
            - Handles both GPOS Format 1 (individual glyph pairs) and Format 2 (class-based)
            - For variable fonts, instantiate with desired axes first:
                ```python
                instance = FontHelper.instantiate_ttfont(variable_font, {"wght": 700})
                kerning = FontHelper.get_kerning_value(instance, "A", "V")
                ```
            - Returns 0 if either character is not in the font
            - Returns 0 if the font has no GPOS table or no kerning data
            - For class-based kerning, determines the class of each glyph and looks up
                the kerning value in the class matrix

        Example:
            >>> font = TTFont("RobotoFlex.ttf")
            >>> kerning = FontHelper.get_kerning_value(font, "A", "V")
            >>> print(f"Kerning for 'AV': {kerning} units")
        """
        cmap = ttfont.getBestCmap()

        # Check if both characters exist
        if ord(left_char) not in cmap or ord(right_char) not in cmap:
            return 0

        left_glyph = cmap[ord(left_char)]
        right_glyph = cmap[ord(right_char)]

        # Check GPOS table for kerning
        if "GPOS" not in ttfont:
            return 0

        gpos = ttfont["GPOS"]

        if hasattr(gpos.table, "LookupList") and gpos.table.LookupList:
            for lookup in gpos.table.LookupList.Lookup:
                if lookup.LookupType == 2:  # PairPos lookup
                    for subtable in lookup.SubTable:
                        if subtable.Format == 1 and hasattr(subtable, "PairSet"):
                            # === Format 1: Individual glyph pairs ===
                            # Find the PairSet for our left glyph
                            for pair_set_idx, pair_set in enumerate(subtable.PairSet):
                                first_glyph = subtable.Coverage.glyphs[pair_set_idx]
                                if first_glyph == left_glyph:
                                    for pair_value_record in pair_set.PairValueRecord:
                                        if pair_value_record.SecondGlyph == right_glyph:
                                            return pair_value_record.Value1.XAdvance if pair_value_record.Value1 else 0

                        elif subtable.Format == 2:
                            # === Format 2: Class-based kerning ===
                            # Get class definitions
                            if hasattr(subtable, "ClassDefRecord") and hasattr(subtable, "Class1Record"):
                                # Find class for left glyph
                                left_class = 0  # Default class 0
                                if hasattr(subtable, "ClassDefRecord"):
                                    for class_idx, class_def in enumerate(subtable.ClassDefRecord):
                                        if hasattr(class_def, "Class") and left_glyph in class_def.Class:
                                            left_class = class_idx
                                            break

                                # Find class for right glyph
                                right_class = 0  # Default class 0
                                if hasattr(subtable, "ClassDefRecord2"):
                                    for class_idx, class_def in enumerate(subtable.ClassDefRecord2):
                                        if hasattr(class_def, "Class") and right_glyph in class_def.Class:
                                            right_class = class_idx
                                            break

                                # Look up kerning value in class matrix
                                if left_class < len(subtable.Class1Record):
                                    class1_record = subtable.Class1Record[left_class]
                                    if hasattr(class1_record, "Class2Record") and right_class < len(
                                        class1_record.Class2Record
                                    ):
                                        class2_record = class1_record.Class2Record[right_class]
                                        if hasattr(class2_record, "Value1") and class2_record.Value1:
                                            return class2_record.Value1.XAdvance if class2_record.Value1.XAdvance else 0

        return 0

    @staticmethod
    def get_ligature(ttfont: TTFont, chars: str) -> str:
        """Get ligature substitution for a character sequence.

        This function looks up ligature substitutions in the GSUB table.
        For example, "ffi" might return "ﬃ" (U+FB03).

        Common ligatures include:
        - "ff", "fi", "fl", "ffi", "ffl"    Common Latin ligatures
                (U+FB00, U+FB01, U+FB02, U+FB03, U+FB04)
        - "ft", "st"
                (U+FB05, U+FB06)
        - "ae", "oe"   Other possible ligatures
                (U+FB07, U+FB08)
        - "IJ", "ij"   Other possible ligatures - Dutch/Icelandic
                (U+FB09, U+FB0A)
        - "ct", "sp"   Special forms in some fonts
                (U+FB0B, U+FB0C)

        Args:
            ttfont (TTFont): A fontTools TTFont object. For variable fonts, this should
                be an instantiated font with specific axis values already applied.
            chars (str): The character sequence to check for ligature (e.g., "ffi").
                Can be 2 or more characters.

        Returns:
            str: The ligature character if found, empty string if no ligature.
                Returns the actual Unicode character, not the glyph name.

        Notes:
            - Handles ligatures of any length (2, 3, 4+ characters)
            - Only handles GSUB LookupType 4 (Ligature Substitution)
            - For variable fonts, ligatures might change based on axis values
            - Returns empty string if no ligature exists
            - If multiple ligatures exist for the same sequence, returns the first one

        Examples:
            >>> font = TTFont("font.ttf")
            >>> lig = FontHelper.get_ligature(font, "ffi")
            >>> print(lig)  # Might print "ﬃ"

            >>> lig = FontHelper.get_ligature(font, "ae")
            >>> print(lig)  # Might print "æ" for fonts that have this ligature
        """
        cmap = ttfont.getBestCmap()

        # Check if all characters exist in font
        for char in chars:
            if ord(char) not in cmap:
                return ""

        # Convert characters to glyph names
        glyph_names = []
        for char in chars:
            glyph_names.append(cmap[ord(char)])

        # Check GSUB table for ligatures
        if "GSUB" not in ttfont:
            return ""

        gsub = ttfont["GSUB"]

        if hasattr(gsub.table, "LookupList") and gsub.table.LookupList:
            for lookup in gsub.table.LookupList.Lookup:
                if lookup.LookupType == 4:  # Ligature substitution
                    for subtable in lookup.SubTable:
                        if hasattr(subtable, "LigatureSet"):
                            # Find LigatureSet for our first glyph
                            for lig_set in subtable.LigatureSet:
                                if lig_set.FirstGlyph == glyph_names[0]:
                                    # Check all ligatures in this set
                                    for ligature in lig_set.Ligature:
                                        # Check if components match our sequence
                                        if ligature.Component == glyph_names[1:]:
                                            # Found a match! Get the ligature glyph
                                            lig_glyph = ligature.LigGlyph

                                            # Convert glyph name back to character
                                            reverse_cmap = {v: k for k, v in cmap.items()}
                                            if lig_glyph in reverse_cmap:
                                                try:
                                                    return chr(reverse_cmap[lig_glyph])
                                                except ValueError:
                                                    # Glyph might be private use or invalid
                                                    return ""

        return ""

    @staticmethod
    def get_number_glyph(ttfont: TTFont, digit: str, tabular: bool = False) -> str:
        """Get a number glyph with tabular or proportional spacing.

        Many fonts provide separate glyphs for numbers with different spacing.
        - Proportional figures: Each number has its natural width (e.g., "1" is narrow, "0" is wide)
        - Tabular figures: All numbers have the same width for perfect alignment in tables

        Args:
            ttfont (TTFont): A fontTools TTFont object.
            digit (str): A single digit character ('0' through '9').
            tabular (bool): If True, returns tabular figure; if False, returns proportional.

        Returns:
            str: The appropriate glyph character, or original digit if no variant exists.

        Notes:
            - Tabular figures often have glyph names ending in '.tf' or '.tnum'
            - Proportional figures may have '.pf' or '.pnum' suffix
            - Some fonts use 'zero.tabular', 'one.tabular', etc.
            - OpenType Feature: 'tnum' for tabular, 'pnum' for proportional
            - If no variant exists, returns the original digit

        Examples:
            >>> font = TTFont("font.ttf")
            >>> # Get tabular zero for tables
            >>> zero_tab = FontHelper.get_number_glyph(font, "0", tabular=True)
            >>> # Get proportional zero for body text
            >>> zero_prop = FontHelper.get_number_glyph(font, "0", tabular=False)
        """
        if len(digit) != 1 or digit not in "0123456789":
            return digit

        cmap = ttfont.getBestCmap()
        reverse_cmap = {v: k for k, v in cmap.items()}

        # Get the base glyph name
        base_code = ord(digit)
        if base_code not in cmap:
            return digit

        base_glyph = cmap[base_code]

        # Common naming patterns for tabular figures
        tabular_suffixes = [
            ".tf",  # Tabular Figure
            ".tnum",  # Tabular Number
            ".tabular",  # Tabular
            ".mono",  # Monospace
        ]

        # Common naming patterns for proportional figures
        proportional_suffixes = [
            ".pf",  # Proportional Figure
            ".pnum",  # Proportional Number
            ".proportional",  # Proportional
        ]

        # Try to find the appropriate variant
        suffixes = tabular_suffixes if tabular else proportional_suffixes

        for suffix in suffixes:
            # Try patterns like "zero.tf", "one.tf", etc.
            variant_glyph = digit + suffix

            # Also try with full names
            digit_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            digit_name = digit_names[int(digit)]
            variant_glyph_full = digit_name + suffix

            # Check if variant exists in cmap
            if variant_glyph in reverse_cmap:
                try:
                    return chr(reverse_cmap[variant_glyph])
                except ValueError:
                    pass

            if variant_glyph_full in reverse_cmap:
                try:
                    return chr(reverse_cmap[variant_glyph_full])
                except ValueError:
                    pass

        # If no variant found, check GSUB for feature substitution
        if "GSUB" in ttfont:
            gsub = ttfont["GSUB"]

            # Look for 'tnum' (tabular) or 'pnum' (proportional) features
            feature_tag = "tnum" if tabular else "pnum"

            if hasattr(gsub.table, "FeatureList") and gsub.table.FeatureList:
                for feature in gsub.table.FeatureList.FeatureRecord:
                    if feature.FeatureTag == feature_tag:
                        # Apply the feature substitution
                        for lookup_idx in feature.Feature.LookupListIndex:
                            if hasattr(gsub.table, "LookupList") and gsub.table.LookupList:
                                lookup = gsub.table.LookupList.Lookup[lookup_idx]
                                if lookup.LookupType == 1:  # Single substitution
                                    for subtable in lookup.SubTable:
                                        if hasattr(subtable, "mapping"):
                                            if base_glyph in subtable.mapping:
                                                target_glyph = subtable.mapping[base_glyph]
                                                if target_glyph in reverse_cmap:
                                                    try:
                                                        return chr(reverse_cmap[target_glyph])
                                                    except ValueError:
                                                        pass

        # No variant found, return original digit
        return digit


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

    _points_buffer: NDArray[np.float64]  # pre-allocated buffer for points
    _point_index: int  # current write position in buffer
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
        # Pre-allocate buffer for points - typical glyphs have 20-100 points
        self._points_buffer: NDArray[np.float64] = np.empty((100, 3), dtype=np.float64)
        self._point_index: int = 0
        # list of command letters (M, L, C, Q, Z)
        self._commands: List[AvGlyphCmds] = []

    def _ensure_capacity(self, additional_points: int):
        """Ensure buffer has capacity for additional points, growing if needed."""
        required_capacity = self._point_index + additional_points
        if required_capacity > len(self._points_buffer):
            # Double the buffer size to amortize growth cost
            new_size = max(required_capacity, len(self._points_buffer) * 2)
            new_buffer = np.empty((new_size, 3), dtype=np.float64)
            new_buffer[: self._point_index] = self._points_buffer[: self._point_index]
            self._points_buffer = new_buffer

    # BasePen callback methods -------------------------------------------------
    def _moveTo(self, pt: Tuple[float, float]):
        self._commands.append("M")
        self._ensure_capacity(1)
        self._points_buffer[self._point_index] = [np.float64(pt[0]), np.float64(pt[1]), np.float64(0.0)]
        self._point_index += 1

    def _line_to_with_type(self, pt: Tuple[float, float], point_type: float):
        self._commands.append("L")
        self._ensure_capacity(1)
        self._points_buffer[self._point_index] = [np.float64(pt[0]), np.float64(pt[1]), np.float64(point_type)]
        self._point_index += 1

    def _lineTo(self, pt: Tuple[float, float]):
        self._line_to_with_type(pt, 0.0)

    def _qCurveToOne(self, pt1: Tuple[float, float], pt2: Tuple[float, float]):
        # quadratic bezier: one control point and an end point
        if self._polygonize_steps > 0:
            self._polygonize_quadratic_bezier([self._getCurrentPoint(), pt1, pt2])
        else:
            self._commands.append("Q")
            self._ensure_capacity(2)
            self._points_buffer[self._point_index] = [np.float64(pt1[0]), np.float64(pt1[1]), np.float64(2.0)]
            self._points_buffer[self._point_index + 1] = [np.float64(pt2[0]), np.float64(pt2[1]), np.float64(0.0)]
            self._point_index += 2

    def _curveToOne(self, pt1: Tuple[float, float], pt2: Tuple[float, float], pt3: Tuple[float, float]):
        # cubic bezier: two control points and an end point
        if self._polygonize_steps > 0:
            self._polygonize_cubic_bezier([self._getCurrentPoint(), pt1, pt2, pt3])
        else:
            self._commands.append("C")
            self._ensure_capacity(3)
            self._points_buffer[self._point_index] = [np.float64(pt1[0]), np.float64(pt1[1]), np.float64(3.0)]
            self._points_buffer[self._point_index + 1] = [np.float64(pt2[0]), np.float64(pt2[1]), np.float64(3.0)]
            self._points_buffer[self._point_index + 2] = [np.float64(pt3[0]), np.float64(pt3[1]), np.float64(0.0)]
            self._point_index += 3

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
        new_points = BezierCurve.polygonize_quadratic_curve(points, self._polygonize_steps)
        # Skip the first point since it's the starting point (already in the path)
        points_to_add = new_points[1:]
        num_points = len(points_to_add)
        self._ensure_capacity(num_points)
        self._points_buffer[self._point_index : self._point_index + num_points] = points_to_add
        self._point_index += num_points
        self._commands.extend(cast(List[AvGlyphCmds], ["L"] * self._polygonize_steps))

    def _polygonize_cubic_bezier(self, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]]):
        new_points = BezierCurve.polygonize_cubic_curve(points, self._polygonize_steps)
        # Skip the first point since it's the starting point (already in the path)
        points_to_add = new_points[1:]
        num_points = len(points_to_add)
        self._ensure_capacity(num_points)
        self._points_buffer[self._point_index : self._point_index + num_points] = points_to_add
        self._point_index += num_points
        self._commands.extend(cast(List[AvGlyphCmds], ["L"] * self._polygonize_steps))

    @property
    def commands(self) -> List[AvGlyphCmds]:
        """Return the recorded commands as a list (uppercase commands)."""
        return self._commands

    @property
    def _points(self) -> NDArray[np.float64]:
        """Return the points array trimmed to actual size."""
        return self._points_buffer[: self._point_index]

    @property
    def points(self) -> NDArray[np.float64]:
        """Return recorded points as an (n_points, 3) ndarray of float64."""
        return self._points

    def reset(self) -> None:
        """Clear recorded commands and points."""
        self._point_index = 0
        self._commands = []


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
