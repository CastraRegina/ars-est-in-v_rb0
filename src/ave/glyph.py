"""Font glyph handling and typography utilities for OpenType and SVG fonts."""

from __future__ import annotations

import gzip
import json
import math
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fontTools.ttLib import TTFont

from ave.common import Align
from ave.font_support import AvFontProperties
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
            return self.bounding_box().width

        raise ValueError(f"Invalid align value: {align}")

    @property
    def height(self) -> float:
        """
        The height of the glyph, i.e. the height of the bounding box (positive value).
        """
        return self.bounding_box().height

    @property
    def ascender(self) -> float:
        """
        The maximum distance above the baseline, i.e. the highest y-coordinate of a glyph (mostly positive value).
        """
        return self.bounding_box().ymax

    @property
    def descender(self) -> float:
        """
        The maximum distance below the baseline, i.e. the lowest y-coordinate of a glyph (usually negative value).
        """
        return self.bounding_box().ymin

    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of a glyph (sign varies +/-).
        Positive values when the glyph is placed to the right of the origin (i.e. positive bounding_box.xmin).
        Negative values when the glyph is placed to the left of the origin (i.e. negative bounding_box.xmin).
        Note: For LEFT or BOTH alignment, this is typically 0.0 as the glyph starts at the origin.
        """
        return self.bounding_box().xmin

    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of a glyph (sign varies +/-).
        Positive values when the glyph's bounding box is inside the advance box (i.e. positive bounding_box.xmax).
        Negative values when the glyph's bounding box extends to the right of the glyph box
                (i.e. bounding_box.xmax > advance_width).
        Note: For RIGHT or BOTH alignment, this is typically 0.0 as the glyph ends at the advance width.
        """
        return self.advance_width - self.bounding_box().xmax

    def bounding_box(self) -> AvBox:
        """
        Returns the tightest bounding box around the glyph's outline.

        Coordinates are relative to baseline-origin (0,0) and use unitsPerEm dimensions.

        Returns:
            AvBox: The bounding box of the glyph's outline.
        """
        # Delegate entirely to AvPath's bounding box implementation
        return self._path.bounding_box()

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

            polygonized: AvSinglePolygonPath = closed_path.polygonized_path()

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
            current_bbox = polygonized.bounding_box()

            for j, other_polygonized in enumerate(polygonized_contours):
                if j != i and other_polygonized is not None:
                    other_area = other_polygonized.area
                    other_bbox = other_polygonized.bounding_box()

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
        bbox = self.bounding_box()
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
# AvGlyphFactory
###############################################################################


class AvGlyphFactory:
    """
    Base class for glyph factories.
    A glyph factory is responsible for creating glyph representations for a character.
    """

    @abstractmethod
    def get_glyph(self, character: str) -> AvGlyph:
        """
        Creates and returns a glyph representation for the specified character.
        Args:
            character (str): The character to create a glyph for.
        Returns:
            AvGlyph: An instance representing the glyph of the specified character.
        """

    @abstractmethod
    def get_font_properties(self) -> AvFontProperties:
        """
        Return font properties.

        Child classes must implement this method to provide font properties.

        Returns:
            AvFontProperties: Font properties
        """


###############################################################################
# AvGlyphCachedFactory
###############################################################################


@dataclass
class AvGlyphCachedFactory(AvGlyphFactory):
    """Base class for glyph factories that cache glyphs.

    This class provides common functionality for caching glyphs and
    serializing/deserializing the cache.
    """

    _glyphs: Dict[str, AvGlyph] = field(init=False)
    _second_source: Optional[AvGlyphFactory] = field(init=False)

    @property
    def glyphs(self) -> Dict[str, AvGlyph]:
        """Return the internal glyph cache."""
        return self._glyphs

    @property
    def second_source(self) -> Optional[AvGlyphFactory]:
        """Return the second source factory."""
        return self._second_source

    @second_source.setter
    def second_source(self, source: Optional[AvGlyphFactory]) -> None:
        """Set the second source factory."""
        self._second_source = source

    def get_glyph(self, character: str) -> AvGlyph:
        """Subclasses must implement this method."""
        raise NotImplementedError("Subclasses must implement get_glyph")

    def get_font_properties(self) -> AvFontProperties:
        """Subclasses must implement this method."""
        raise NotImplementedError("Subclasses must implement get_font_properties")

    def to_cache_dict(self) -> dict:
        """
        Create a dictionary representation of the cached glyphs.

        Returns:
            dict: Dictionary with "glyphs" and "font_properties".
        """
        glyphs_dict: Dict[str, dict] = {character: glyph.to_dict() for character, glyph in self._glyphs.items()}
        return {
            "format_version": 1,
            "type": self.__class__.__name__,
            "characters": "".join(sorted(glyphs_dict.keys())),
            "glyphs": glyphs_dict,
            "font_properties": self.get_font_properties().to_dict(),
        }

    @classmethod
    def from_cache_dict(cls, data: dict, second_source: Optional[AvGlyphFactory] = None):
        """
        Create a factory from cache data dictionary.

        Args:
            data: Dictionary with "glyphs" and "font_properties".
            second_source: Optional second source factory to set after loading.

        Returns:
            Factory instance with loaded glyphs.

        Raises:
            ValueError: If data type doesn't match class name.
        """
        if data.get("type") != cls.__name__:
            raise ValueError(f"Invalid data type: expected '{cls.__name__}', got '{data.get('type')}'")

        glyph_entries = data.get("glyphs", {})
        glyphs: Dict[str, AvGlyph] = {}

        for character, glyph_data in glyph_entries.items():
            try:
                glyphs[character] = AvGlyph.from_dict(glyph_data)
            except (ValueError, KeyError, TypeError) as e:
                print(f"Warning: Failed to load glyph for '{character}': {e}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error loading glyph '{character}': {e}") from e

        # Create factory without calling __init__ (subclasses will handle it)
        factory = cls.__new__(cls)
        factory._glyphs = glyphs
        factory._second_source = second_source
        return factory

    def save_to_file(self, file_path: str) -> None:
        """Save factory to compressed JSON file."""
        cache_data = self.to_cache_dict()
        target_path = Path(file_path)

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as gzipped JSON
        with gzip.open(target_path, "wt", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: str, second_source: Optional[AvGlyphFactory] = None):
        """Load factory from compressed JSON file.

        Args:
            file_path: Path to the compressed JSON file.
            second_source: Optional second source factory to set after loading.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {file_path}")

        try:
            # Try gzip first
            with gzip.open(path, "rt", encoding="utf-8") as f:
                cache_data = json.load(f)
        except (gzip.BadGzipFile, OSError):
            # Fall back to regular JSON
            with open(path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in cache file {file_path}: {e}") from e

        return cls.from_cache_dict(cache_data, second_source)


###############################################################################
# AvGlyphCachedSourceFactory
###############################################################################


@dataclass
class AvGlyphCachedSourceFactory(AvGlyphCachedFactory):
    """Glyph factory with cache and fallback to second source.

    This factory maintains an internal glyph cache. When get_glyph() is called,
    it first checks the cache, and if not found, retrieves from the second source
    and caches the result for future requests.

    Overview:
    - glyphs: {} or filled (cached during operation)
    - _second_source: mandatory (required for font properties and glyph generation)
    - font_properties: provided by _second_source
    """

    def __init__(
        self,
        second_source: AvGlyphFactory,
        glyphs: Optional[Dict[str, AvGlyph]] = None,
    ) -> None:
        """Initialize the factory with a second source and optional initial cache.

        Args:
            second_source: Factory to use when glyph is not in cache.
            glyphs: Optional initial glyph cache.
        """
        self._second_source = second_source  # Must not be None
        self._glyphs = glyphs if glyphs is not None else {}

    def get_font_properties(self) -> AvFontProperties:
        """Return font properties from the second source."""
        return self._second_source.get_font_properties()

    def get_glyph(self, character: str) -> AvGlyph:
        """
        Retrieve a glyph from cache or second source.

        First checks if the glyph is in the internal cache. If not found,
        retrieves it from the second source, caches it, and returns it.

        Args:
            character (str): The character to retrieve.

        Returns:
            AvGlyph: The glyph instance from cache or second source.

        Raises:
            KeyError: If glyph is not found in second source.
        """
        # Check cache first
        if character in self._glyphs:
            return self._glyphs[character]

        # Get from second source and cache it
        glyph = self._second_source.get_glyph(character)
        self._glyphs[character] = glyph
        return glyph

    @property
    def second_source(self) -> AvGlyphFactory:
        """Return the second source factory (never None in this class)."""
        return self._second_source

    @second_source.setter
    def second_source(self, source: AvGlyphFactory) -> None:
        """Set the second source factory (must not be None)."""
        if source is None:
            raise ValueError("AvGlyphCachedSourceFactory requires a non-None second_source")
        self._second_source = source

    @classmethod
    def from_cache_dict(
        cls, data: dict, second_source: Optional[AvGlyphFactory] = None
    ) -> "AvGlyphCachedSourceFactory":
        """
        Create a cached factory from cache data dictionary.

        Args:
            data: Dictionary with "glyphs" and "font_properties".
            second_source: Second source factory to set after loading (required for this class).

        Returns:
            AvGlyphCachedSourceFactory: Factory with loaded glyphs.

        Raises:
            ValueError: If data type is not "AvGlyphCachedSourceFactory".
            ValueError: If second_source is None (required for this class).
        """
        if data.get("type") != "AvGlyphCachedSourceFactory":
            raise ValueError(f"Invalid data type: expected 'AvGlyphCachedSourceFactory', got '{data.get('type')}'")

        if second_source is None:
            raise ValueError("AvGlyphCachedSourceFactory requires a non-None second_source")

        glyph_entries = data.get("glyphs", {})
        glyphs: Dict[str, AvGlyph] = {}

        for character, glyph_data in glyph_entries.items():
            try:
                glyphs[character] = AvGlyph.from_dict(glyph_data)
            except (ValueError, KeyError, TypeError) as e:
                print(f"Warning: Failed to load glyph for '{character}': {e}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error loading glyph '{character}': {e}") from e

        # Create factory without calling __init__ (parent class handles it)
        factory = cls.__new__(cls)
        factory._glyphs = glyphs
        factory._second_source = second_source
        return factory


###############################################################################
# AvGlyphPersistentFactory
###############################################################################


@dataclass
class AvGlyphPersistentFactory(AvGlyphCachedFactory):
    """Glyph factory that loads/saves glyphs from/to persistent storage.

    This factory is initialized with glyphs and font properties from a file
    or other source, and optionally has a second source for generating new glyphs.

    Overview:
    - glyphs: {} or filled (input from dict/file)
    - _second_source: available or None (optional for generating new glyphs)
    - font_properties: provided by _second_source if available, otherwise read from dict/file
    """

    _font_properties: AvFontProperties = field(init=False)  # Own font properties

    def __init__(
        self,
        glyphs: Dict[str, AvGlyph],
        font_properties: AvFontProperties,
        second_source: Optional[AvGlyphFactory] = None,
    ) -> None:
        """Initialize the factory with glyphs and font properties.

        Args:
            glyphs: Initial glyph dictionary.
            font_properties: Font properties.
            second_source: Optional factory for generating new glyphs.
        """
        self._glyphs = glyphs
        self._font_properties = font_properties
        self._second_source = second_source

    def get_font_properties(self) -> AvFontProperties:
        """Return font properties from second source or own properties."""
        if self._second_source is not None:
            return self._second_source.get_font_properties()
        return self._font_properties

    def get_glyph(self, character: str) -> AvGlyph:
        """
        Retrieve a glyph from cache or second source.

        First checks if the glyph is in the internal cache. If not found
        and a second source is available, retrieves it from the second source
        and caches it.

        Args:
            character (str): The character to retrieve.

        Returns:
            AvGlyph: The glyph instance from cache or second source.

        Raises:
            KeyError: If glyph is not found in cache and no second source is set.
        """
        # Check cache first
        if character in self._glyphs:
            return self._glyphs[character]

        # Check if we have a second source
        if self._second_source is None:
            raise KeyError(f"Glyph for character {character!r} not found in cache and no second source is set")

        # Get from second source and cache it
        glyph = self._second_source.get_glyph(character)
        self._glyphs[character] = glyph
        return glyph

    @classmethod
    def from_cache_dict(cls, data: dict, second_source: Optional[AvGlyphFactory] = None) -> "AvGlyphPersistentFactory":
        """
        Create a persistent factory from cache data dictionary.

        Args:
            data: Dictionary with "glyphs" and "font_properties".
            second_source: Optional second source factory to set after loading.

        Returns:
            AvGlyphPersistentFactory: Factory with loaded glyphs and properties.

        Raises:
            ValueError: If data type is not "AvGlyphPersistentFactory".
        """
        if data.get("type") != cls.__name__:
            raise ValueError(f"Invalid data type: expected '{cls.__name__}', got '{data.get('type')}'")

        glyph_entries = data.get("glyphs", {})
        glyphs: Dict[str, AvGlyph] = {}

        for character, glyph_data in glyph_entries.items():
            try:
                glyphs[character] = AvGlyph.from_dict(glyph_data)
            except (ValueError, KeyError, TypeError) as e:
                print(f"Warning: Failed to load glyph for '{character}': {e}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error loading glyph '{character}': {e}") from e

        font_properties = AvFontProperties.from_dict(data.get("font_properties", {}))

        # Create factory without calling __init__ (parent class handles it)
        factory = cls.__new__(cls)
        factory._glyphs = glyphs
        factory._font_properties = font_properties
        factory._second_source = second_source
        return factory


###############################################################################
# AvGlyphFromTTFontFactory
###############################################################################


@dataclass
class AvGlyphFromTTFontFactory(AvGlyphFactory):
    """Factory class for creating glyph instances."""

    _ttfont: TTFont = field(init=False)

    def __init__(self, ttfont: TTFont) -> None:
        """
        Initializes the glyph factory.
        """
        # No super().__init__() needed - no parent class initialization required
        self._ttfont = ttfont

    @property
    def ttfont(self) -> TTFont:
        """
        Returns the TTFont instance associated with this glyph factory.
        """
        return self._ttfont

    def get_glyph(self, character: str) -> AvGlyph:
        return AvGlyph.from_ttfont_character(self._ttfont, character)

    def get_font_properties(self) -> AvFontProperties:
        """
        Return font properties extracted from the TTFont.

        Returns:
            AvFontProperties: Font properties from TTFont
        """
        return AvFontProperties.from_ttfont(self._ttfont)


###############################################################################
# AvGlyphPolygonizeFactory
###############################################################################


@dataclass
class AvGlyphPolygonizeFactory(AvGlyphFactory):
    """
    A glyph factory wrapper that polygonizes glyphs from another factory.

    It retrieves glyphs from a source factory, polygonizes their paths,
    and returns new AvGlyph instances with polygonized paths.
    """

    _source_factory: AvGlyphFactory = field(init=False)
    _polygonize_steps: int = field(init=False)

    def __init__(
        self,
        source_factory: AvGlyphFactory,
        polygonize_steps: int = 50,
    ) -> None:
        """
        Initialize the polygonizing factory.

        Parameters:
            source_factory (AvGlyphFactory): The factory to retrieve glyphs from.
            polygonize_steps (int, optional): Number of steps for polygonization.
                Defaults to 50. 0 = no polygonization
        """
        # No super().__init__() needed - no parent class initialization required
        self._source_factory = source_factory
        self._polygonize_steps = polygonize_steps

    @property
    def source_factory(self) -> AvGlyphFactory:
        """Return the underlying source glyph factory."""
        return self._source_factory

    @property
    def polygonize_steps(self) -> int:
        """Number of steps used for polygonization."""
        return self._polygonize_steps

    @source_factory.setter
    def source_factory(self, source_factory: AvGlyphFactory) -> None:
        self._source_factory = source_factory

    @polygonize_steps.setter
    def polygonize_steps(self, polygonize_steps: int) -> None:
        self._polygonize_steps = polygonize_steps

    def get_glyph(self, character: str) -> AvGlyph:
        """
        Retrieve a glyph from the source factory, polygonize its path,
        and return a new AvGlyph with the polygonized path.

        Args:
            character (str): The character to create a polygonized glyph for.

        Returns:
            AvGlyph: A new glyph instance with a polygonized path.
        """
        # Retrieve the original glyph from the source factory
        original_glyph = self._source_factory.get_glyph(character)

        # Polygonize the path
        polygonized_path = original_glyph.path.polygonize(self._polygonize_steps)

        # Return a new AvGlyph with the polygonized path
        return AvGlyph(
            character=original_glyph.character,
            advance_width=original_glyph.width(),
            path=polygonized_path,
        )

    def get_font_properties(self) -> AvFontProperties:
        """
        Return font properties from the source factory.

        Forwards the request to the source factory since polygonization
        doesn't affect font properties.

        Returns:
            AvFontProperties: Font properties from source factory
        """
        return self._source_factory.get_font_properties()


###############################################################################
# AvGlyphDualSourceFactory
###############################################################################


@dataclass
class AvGlyphDualSourceFactory(AvGlyphFactory):
    """Glyph factory that tries two sources in sequence.

    The factory first tries to get a glyph from the first source.
    If the first source raises an exception, it tries the second source.
    """

    _first_source: AvGlyphFactory = field(init=False)
    _second_source: AvGlyphFactory = field(init=False)
    _font_props_secondary: bool = field(init=False)  # True to use second source for font properties

    def __init__(
        self,
        first_source: AvGlyphFactory,
        second_source: AvGlyphFactory,
        font_props_secondary: bool = True,
    ) -> None:
        """Initialize the factory with two sources.

        Args:
            first_source: Primary glyph factory to try first.
            second_source: Secondary glyph factory to try if first fails.
            font_props_secondary: If True, use second source for font properties.
        """
        self._first_source = first_source
        self._second_source = second_source
        self._font_props_secondary = font_props_secondary

    def get_font_properties(self) -> AvFontProperties:
        """Return font properties from the configured source."""
        if self._font_props_secondary:
            return self._second_source.get_font_properties()
        return self._first_source.get_font_properties()

    def get_glyph(self, character: str) -> AvGlyph:
        """
        Retrieve a glyph from either source.

        First tries the first source. If it raises an exception,
        tries the second source.

        Args:
            character (str): The character to retrieve.

        Returns:
            AvGlyph: The glyph instance from one of the sources.

        Raises:
            KeyError: If glyph is not found in either source.
        """
        try:
            return self._first_source.get_glyph(character)
        except KeyError:
            # First source failed, try second source
            return self._second_source.get_glyph(character)


###############################################################################
# Main
###############################################################################


def main():
    """Main"""


if __name__ == "__main__":
    main()
