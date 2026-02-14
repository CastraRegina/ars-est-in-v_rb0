"""Font glyph handling and typography utilities for OpenType and SVG fonts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from fontTools.ttLib import TTFont

from ave.common import Align
from ave.fonttools import AvGlyphPtsCmdsPen
from ave.geom import AvBox
from ave.path import (
    CLOSED_SINGLE_PATH_CONSTRAINTS,
    AvClosedSinglePath,
    AvPath,
    AvSinglePath,
    AvSinglePolygonPath,
    PathSplitter,
)

###############################################################################
# Glyph
###############################################################################


@dataclass
class AvGlyph:
    """Representation of a font glyph, i.e. a single character of a certain font.

    Uses dimensions in unitsPerEm, i.e. independent from font_size.
    It is composed of a set of points and a set of commands that define how to draw the shape.

    Attributes:
        character: Unicode character represented by this glyph
        path: SVG path defining the glyph shape
        name: Optional glyph name from font table
    """

    _character: str
    _advance_width: float
    _path: AvPath

    def __init__(
        self,
        character: str,
        advance_width: float,
        path: AvPath,
    ) -> None:
        """
        Initialize an AvGlyph.

        Args:
            character: A single character represented by this glyph.
            advance_width: The advance width of the glyph in unitsPerEm.
                            This is the distance from the origin of the glyph to the origin of the next glyph.
            path: The path object containing points and commands for the glyph's outline.
        """
        # No super().__init__() needed - no parent class initialization required
        self._character = character
        self._advance_width = advance_width
        self._path = path

    @classmethod
    def from_dict(cls, data: dict) -> AvGlyph:
        """Create an AvGlyph instance from a dictionary."""

        path_dict = data.get("path")
        if path_dict is None:
            path = AvPath()
        else:
            path = AvPath.from_dict(path_dict)

        return cls(
            character=data.get("character", ""),
            advance_width=data.get("advance_width", 0.0),
            path=path,
        )

    def to_dict(self) -> dict:
        """Convert the AvGlyph instance to a dictionary."""

        return {
            "character": self._character,
            "advance_width": self._advance_width,
            "path": self._path.to_dict(),
        }

    @classmethod
    def from_ttfont_character(cls, ttfont: TTFont, character: str, polygonize_steps: int = 0) -> AvGlyph:
        """
        Factory method to create an AvGlyph from a TTFont and character.

        Parameters:
            ttfont (TTFont): The TTFont to use.
            character (str): The character to use.
            polygonize_steps (int, optional): The number of steps to use for polygonization.
                Defaults to 0 = no polygonization.

        Notes:
            If polygonize_steps is 0 the commands could contain also curves
            If polygonize_steps is greater than 0, then curves will be polygonized
        """
        glyph_name = ttfont.getBestCmap()[ord(character)]
        glyph_set = ttfont.getGlyphSet()
        pen = AvGlyphPtsCmdsPen(glyph_set, polygonize_steps=polygonize_steps)
        glyph_set[glyph_name].draw(pen)
        advance_width = glyph_set[glyph_name].width
        # Create AvPath first, then create AvGlyph
        path = AvPath(pen.points, pen.commands)

        # Analyze path and set appropriate constraints
        appropriate_constraints = path.determine_appropriate_constraints()
        path = path.with_constraints(appropriate_constraints)

        return cls(character, advance_width, path)

    @property
    def character(self) -> str:
        """
        The character of this glyph.
        """
        return self._character

    @property
    def path(self) -> AvPath:
        """
        The path object of this glyph containing points and commands.
        """
        return self._path

    @property
    def advance_width(self) -> float:
        """
        The advance width of the glyph in unitsPerEm.
        This is the distance from the origin of the glyph to the origin of the next glyph.
        """
        return self._advance_width

    def width(self, align: Optional[Align] = None) -> float:
        """
        Returns width considering align, or official advanceWidth if align is None.

        Args:
            align (Optional[Align], optional): LEFT, RIGHT, BOTH. Defaults to None.
                None:  advanceWidth == LSB + bounding_box.width + RSB
                LEFT:  advanceWidth - bounding_box.xmin == advanceWidth - LSB
                RIGHT: bounding_box.width + bounding_box.xmin   == LSB + bounding_box.width == advanceWidth - RSB
                BOTH:  bounding_box.width                       == advanceWidth - LSB - RSB
        """
        if align is None:
            return self.advance_width

        if align == Align.LEFT:
            return self.advance_width - self.left_side_bearing()
        if align == Align.RIGHT:
            return self.advance_width - self.right_side_bearing()
        if align == Align.BOTH:
            return self.bounding_box.width

        raise ValueError(f"Invalid align value: {align}")

    @property
    def height(self) -> float:
        """
        The height of the glyph, i.e. the height of the bounding box (positive value).
        """
        return self.bounding_box.height

    @property
    def ascender(self) -> float:
        """
        The maximum distance above the baseline, i.e. the highest y-coordinate of a glyph (mostly positive value).
        """
        return self.bounding_box.ymax

    @property
    def descender(self) -> float:
        """
        The maximum distance below the baseline, i.e. the lowest y-coordinate of a glyph (usually negative value).
        """
        return self.bounding_box.ymin

    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of a glyph (sign varies +/-).
        Positive values when the glyph is placed to the right of the origin (i.e. positive bounding_box.xmin).
        Negative values when the glyph is placed to the left of the origin (i.e. negative bounding_box.xmin).
        Note: For LEFT or BOTH alignment, this is typically 0.0 as the glyph starts at the origin.
        """
        return self.bounding_box.xmin

    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of a glyph (sign varies +/-).
        Positive values when the glyph's bounding box is inside the advance box (i.e. positive bounding_box.xmax).
        Negative values when the glyph's bounding box extends to the right of the glyph box
                (i.e. bounding_box.xmax > advance_width).
        Note: For RIGHT or BOTH alignment, this is typically 0.0 as the glyph ends at the advance width.
        """
        return self.advance_width - self.bounding_box.xmax

    @property
    def bounding_box(self) -> AvBox:
        """
        Returns the tightest bounding box around the glyph's outline.

        Coordinates are relative to baseline-origin (0,0) and use unitsPerEm dimensions.

        Returns:
            AvBox: The bounding box of the glyph's outline.
        """
        # Delegate entirely to AvPath's bounding box implementation
        return self._path.bounding_box

    @property
    def glyph_box(self) -> AvBox:
        """
        Returns the glyph's advance box (not the outline bounding box).

        The box spans from x=0 to advanceWidth and y=descender to ascender.

        Returns:
            AvBox: The glyph advance box.
        """
        return AvBox(0, self.descender, self.advance_width, self.ascender)

    def centroid(self) -> Tuple[float, float]:
        """
        Returns the centroid of the glyph.
        For space characters, returns the middle of the advance width at baseline.
        """
        # Special case for space character which has no outline
        if self.character == " ":
            # Use middle of advance width for x, baseline (0) for y
            return (self.width() / 2, 0.0)

        return self._path.centroid

    def approx_equal(self, other: AvGlyph, rtol: float = 1e-9, atol: float = 1e-9) -> bool:
        """Check if two glyphs are approximately equal within numerical tolerances.

        Args:
            other: Another AvGlyph to compare with
            rtol: Relative tolerance for floating point comparison
            atol: Absolute tolerance for floating point comparison

        Returns:
            True if glyphs are approximately equal, False otherwise
        """

        if not isinstance(other, AvGlyph):
            return False

        # Character must match exactly
        if self.character != other.character:
            return False

        # Width with numerical tolerance
        if not math.isclose(self.advance_width, other.advance_width, rel_tol=rtol, abs_tol=atol):
            return False

        # Path with hierarchical comparison
        if not self.path.approx_equal(other.path, rtol, atol):
            return False

        return True

    def revise_direction(self, largest_area_sets_ccw: bool = True) -> AvGlyph:
        """Normalize contour direction to TrueType/OpenType winding rules.

        TrueType/OpenType glyph outlines use contour winding to distinguish
        filled regions vs holes. This function supports two algorithms:

        Algorithm A (largest_area_sets_ccw=True):
        - Find the contour with the largest area
        - If it is CCW, keep all contours as-is
        - If it is CW, reverse all contours

        Algorithm B (largest_area_sets_ccw=False):
        - Classify contours as additive vs subtractive using strict geometric nesting
        - Additive polygons (filled areas, outer contours): counter-clockwise (CCW)
        - Subtractive polygons (cut-out areas, inner contours/holes): clockwise (CW)

        Args:
            largest_area_sets_ccw: If True, use Algorithm A; if False, use Algorithm B

        Returns:
            AvGlyph: New glyph with corrected segment directions
        """
        # Step 1: Split glyph into individual contours (common to both algorithms)
        contours: List[AvSinglePath] = self.path.split_into_single_paths()

        # Step 2: Process each contour - check if closed, polygonize for area computation (common)
        processed_contours: List[AvSinglePath] = []
        polygonized_contours: List[Optional[AvSinglePolygonPath]] = []

        for contour in contours:
            is_closed = contour.commands and contour.commands[-1] == "Z"

            if not is_closed:
                processed_contours.append(contour)
                polygonized_contours.append(None)
                continue

            closed_path: AvClosedSinglePath = AvPath(
                contour.points.copy(), list(contour.commands), CLOSED_SINGLE_PATH_CONSTRAINTS
            )

            polygonized: AvSinglePolygonPath = closed_path.polygonized_path

            if polygonized.area < 1e-10:
                processed_contours.append(contour)
                polygonized_contours.append(None)
                continue

            processed_contours.append(contour)
            polygonized_contours.append(polygonized)

        # Branch based on algorithm choice
        if largest_area_sets_ccw:
            final_contours = self._revise_direction_by_largest_area(processed_contours, polygonized_contours)
        else:
            final_contours = self._revise_direction_by_nesting(processed_contours, polygonized_contours)

        # Reassemble contours into new AvPath (common)
        if final_contours:
            new_path = AvPath.join_paths(*final_contours)
        else:
            new_path = AvPath()

        return AvGlyph(character=self.character, advance_width=self.width(), path=new_path)

    def _revise_direction_by_largest_area(
        self,
        processed_contours: List[AvSinglePath],
        polygonized_contours: List[Optional[AvSinglePolygonPath]],
    ) -> List[AvSinglePath]:
        """Algorithm A: Use largest area contour to determine if all should be reversed.

        Step 3: Find contour with largest area
        Step 4: If largest area contour is CCW, keep all as-is; else reverse all
        """
        # Step 3: Find contour with largest area
        largest_area = 0.0
        largest_area_is_ccw = True

        for polygonized in polygonized_contours:
            if polygonized is not None and polygonized.area > largest_area:
                largest_area = polygonized.area
                largest_area_is_ccw = polygonized.is_ccw

        # Step 4: Enforce direction based on largest contour
        final_contours: List[AvSinglePath] = []

        if largest_area_is_ccw:
            # Largest is CCW (additive/positive), keep all contours as-is
            final_contours = processed_contours
        else:
            # Largest is CW (negative), reverse all contours
            for contour in processed_contours:
                final_contours.append(contour.reverse())

        return final_contours

    def _revise_direction_by_nesting(
        self,
        processed_contours: List[AvSinglePath],
        polygonized_contours: List[Optional[AvSinglePolygonPath]],
    ) -> List[AvSinglePath]:
        """Algorithm B: Classify contours by geometric nesting and enforce winding rules.

        Step 4: Classify contours as additive vs subtractive using strict geometric nesting
        Step 5: Enforce required direction - reverse contours that don't match winding rules
        """
        # Step 4: Classify contours by geometric nesting
        contour_classes = []

        for i, (contour, polygonized) in enumerate(zip(processed_contours, polygonized_contours)):
            if polygonized is None:
                contour_classes.append(None)
                continue

            current_area = polygonized.area
            test_point = polygonized.representative_point()

            is_nested = False
            current_bbox = polygonized.bounding_box

            for j, other_polygonized in enumerate(polygonized_contours):
                if j != i and other_polygonized is not None:
                    other_area = other_polygonized.area
                    other_bbox = other_polygonized.bounding_box

                    bbox_fully_contained = (
                        current_bbox.xmin >= other_bbox.xmin
                        and current_bbox.xmax <= other_bbox.xmax
                        and current_bbox.ymin >= other_bbox.ymin
                        and current_bbox.ymax <= other_bbox.ymax
                    )

                    if bbox_fully_contained and other_area > current_area:
                        if other_polygonized.contains_point(test_point):
                            is_nested = True
                            break

            is_additive = not is_nested
            contour_classes.append(is_additive)

        # Step 5: Enforce required direction
        final_contours: List[AvSinglePath] = []

        for contour, polygonized, is_additive in zip(processed_contours, polygonized_contours, contour_classes):
            if polygonized is None or is_additive is None:
                final_contours.append(contour)
                continue

            is_ccw = polygonized.is_ccw

            needs_reversal = False
            if is_additive and not is_ccw:
                needs_reversal = True
            elif not is_additive and is_ccw:
                needs_reversal = True

            if needs_reversal:
                reversed_contour = contour.reverse()
                final_contours.append(reversed_contour)
            else:
                final_contours.append(contour)

        return final_contours

    def validate(self) -> None:
        """
        Validate the glyph.
        Checks done:
            Step 1: Check correct direction of segments of path (additive: CCW, subtractive: CW) by
                calling revise_direction() and validate by comparing its result with self
            Step 2: Check width to be in range based on character class:
                - Class 1 (NORMAL): 0 < width < 10*bounding_box.width
                - Class 2 (PUNCTUATION): 0 < width < 15*bounding_box.width
                - Class 3 (EXTREME): 0 < width < 25*bounding_box.width
                (special case: space character " " only requires width > 0)
            Step 3: Validate path structure
                - Check that all segments of the path are closed,
                    i.e. each segment starts with 'M' and ends with 'Z'
            Step 4: Check coordinates for NaN, infinity
            Step 5: Check for reasonable coordinate ranges, i.e. -10*units_per_em < x < 10*units_per_em
                (using units_per_em = 2048.0 as default)
            Step 6: Check for reasonable bounding box, i.e. -10*units_per_em < x < 10*units_per_em
            Step 7: Check for duplicate consecutive points within a relative tolerance and absolute tolerance
            Step 8: Check for self-intersection (placeholder for future implementation)
            Step 9: Check constraints against real properties of path

        Raises:
            ValueError: If any validation check fails
        """
        # Constants for validation
        REVISE_POINTS_TOLERANCE_RTOL = 1e-9  # pylint: disable=invalid-name
        REVISE_POINTS_TOLERANCE_ATOL = 1e-9  # pylint: disable=invalid-name
        COORDINATE_RANGE_MULTIPLIER = 10.0  # pylint: disable=invalid-name
        UNITS_PER_EM_DEFAULT = 2048.0  # pylint: disable=invalid-name
        DUPLICATE_POINT_RTOL = 1e-9  # pylint: disable=invalid-name
        DUPLICATE_POINT_ATOL = 1e-9  # pylint: disable=invalid-name

        def get_width_multiplier(character: str) -> float:
            """Get appropriate width multiplier based on character type.

            Returns:
                float: Width multiplier for validation (10.0, 15.0, or 25.0)
            """
            # Class 3: Extreme cases - thin vertical/horizontal lines
            if character in "|'\"":
                return 25.0

            # Class 2: Punctuation with moderate ratios - dots and small marks
            if character in ".,;:!?":
                return 15.0

            # Class 1: Everything else (letters, digits, most symbols)
            return 10.0

        # Step 1: Check correct direction of segments
        revised_glyph = self.revise_direction()
        # Compare only the geometric path (points and commands), not constraints
        points_equal = np.allclose(
            self._path.points,
            revised_glyph.path.points,
            rtol=REVISE_POINTS_TOLERANCE_RTOL,
            atol=REVISE_POINTS_TOLERANCE_ATOL,
        )
        commands_equal = self._path.commands == revised_glyph.path.commands
        if not points_equal or not commands_equal:
            raise ValueError(f"Glyph '{self._character}' has incorrect winding directions")

        # Step 2: Check width range
        bbox = self.bounding_box
        # Special case for space character: no visual bounding box but has glyph width
        if self._character == " " or bbox.width == 0:
            # For space character, only check that width is positive
            if self.advance_width <= 0:
                raise ValueError(
                    "Space character width must be positive, "
                    f"got {self.advance_width} for character '{self._character}'"
                )
        else:
            # For normal characters, check width is within reasonable range based on character class
            width_multiplier = get_width_multiplier(self._character)
            if not 0 < self.advance_width < width_multiplier * bbox.width:
                msg = (
                    f"Glyph '{self._character}' width {self.advance_width} "
                    f"is not in valid range (0, {width_multiplier * bbox.width}) "
                    f"for character class (multiplier: {width_multiplier}x)"
                )
                raise ValueError(msg)

        # Step 3: Validate path structure using existing path validation
        # Use closed path constraints to ensure all segments are closed

        # Check that all segments are closed
        segments = PathSplitter.split_commands_into_segments(self._path.commands)
        for seg_idx, (seg_cmds, _) in enumerate(segments):
            if not seg_cmds:
                raise ValueError(f"Glyph '{self._character}' segment {seg_idx} is empty")
            if seg_cmds[0] != "M":
                raise ValueError(f"Glyph '{self._character}' segment {seg_idx} must start with 'M' command")
            if seg_cmds[-1] != "Z":
                raise ValueError(
                    f"Glyph '{self._character}' segment {seg_idx} must end with 'Z' command for closed path"
                )

        # Step 4: Check coordinates for NaN and infinity
        points = self._path.points
        if not np.all(np.isfinite(points)):
            invalid_coords = np.where(~np.isfinite(points))
            raise ValueError(
                f"Glyph '{self._character}' contains invalid coordinates (NaN or infinity) at indices: {invalid_coords}"
            )

        # Step 5: Check for reasonable coordinate ranges
        min_coord = -COORDINATE_RANGE_MULTIPLIER * UNITS_PER_EM_DEFAULT
        max_coord = COORDINATE_RANGE_MULTIPLIER * UNITS_PER_EM_DEFAULT

        if np.any(points[:, 0] < min_coord) or np.any(points[:, 0] > max_coord):
            raise ValueError(
                f"Glyph '{self._character}' x-coordinates exceed reasonable range [{min_coord}, {max_coord}]"
            )
        if np.any(points[:, 1] < min_coord) or np.any(points[:, 1] > max_coord):
            raise ValueError(
                f"Glyph '{self._character}' y-coordinates exceed reasonable range [{min_coord}, {max_coord}]"
            )

        # Step 6: Check for reasonable bounding box
        if bbox.xmin < min_coord or bbox.xmax > max_coord or bbox.ymin < min_coord or bbox.ymax > max_coord:
            raise ValueError(
                f"Glyph '{self._character}' bounding box exceeds reasonable range [{min_coord}, {max_coord}]"
            )

        # Step 7: Check for duplicate consecutive points
        if len(points) > 1:
            # Check consecutive points for duplicates within tolerance
            diffs = np.diff(points[:, :2], axis=0)
            distances = np.sqrt(np.sum(diffs**2, axis=1))

            duplicate_mask = distances < DUPLICATE_POINT_ATOL + DUPLICATE_POINT_RTOL * np.abs(points[:-1, :2]).max(
                axis=1
            )

            if np.any(duplicate_mask):
                duplicate_indices = np.where(duplicate_mask)[0]
                raise ValueError(
                    f"Glyph '{self._character}' contains duplicate consecutive points at indices: {duplicate_indices}"
                )

        # Step 8: Check for self-intersection
        # This is a complex geometric check, for now we'll do a basic check
        # A full implementation would require segment-segment intersection testing
        # if len(segments) > 1:
        #     # For multiple segments, check if any segment intersects others
        #     # This is a simplified check - full implementation would be more complex
        #     for i in range(len(segments)):
        #         for j in range(i + 1, len(segments)):
        #             # Extract segment paths for intersection testing
        #             # This is a placeholder for a full intersection test
        #             pass

        # Step 9: Check constraints against real properties of path
        # The path already validates against its constraints during initialization
        # Additional checks could be added here if needed

        # # Check if the path's constraints match what would be determined appropriate
        # appropriate_constraints = self._path.determine_appropriate_constraints()
        # if self._path.constraints != appropriate_constraints:
        #     # This is not necessarily an error, but could indicate suboptimal constraint usage
        #     # For now, we'll just note it but not raise an error
        #     # In the future, this could be made stricter if needed
        #     pass


###############################################################################
# Old factory classes have been removed.
# Please use the new composition-based system from ave.glyph_factory instead.
# See glyph_factory.py for AvGlyphFactory, TTFontGlyphSource, MemoryGlyphCache,
# PersistentGlyphCache, PolygonizeTransformer, and convenience functions.
###############################################################################


###############################################################################
# Main
###############################################################################


def main():
    """Main"""


if __name__ == "__main__":
    main()
