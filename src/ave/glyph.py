"""Font glyph handling and typography utilities for OpenType and SVG fonts."""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from fontTools.ttLib import TTFont
from numpy.typing import NDArray

import ave.common
from ave.common import AvGlyphCmds
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
    _width: float
    _path: AvPath

    def __init__(
        self,
        character: str,
        width: float,
        path: AvPath,
    ) -> None:
        """
        Initialize an AvGlyph.

        Args:
            character (str): A single character.
            width (float): The width of the glyph in unitsPerEm.
            path (AvPath): The path object containing points and commands for the glyph.
        """
        super().__init__()
        self._character = character
        self._width = width
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
            width=data.get("width", 0.0),
            path=path,
        )

    def to_dict(self) -> dict:
        """Convert the AvGlyph instance to a dictionary."""

        return {
            "character": self._character,
            "width": self._width,
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
        width = glyph_set[glyph_name].width
        # Create AvPath first, then create AvGlyph
        path = AvPath(pen.points, pen.commands)

        # Analyze path and set appropriate constraints
        appropriate_constraints = path.determine_appropriate_constraints()
        path = path.with_constraints(appropriate_constraints)

        return cls(character, width, path)

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

    def width(self, align: Optional[ave.common.Align] = None) -> float:
        """
        Returns the width calculated considering the alignment.
        Returns the official glyph_width of this glyph if align is None.

        Args:
            align (Optional[av.consts.Align], optional): LEFT, RIGHT, BOTH. Defaults to None.
                None:  official glyph_width (i.e. including LSB and RSB)
                LEFT:  official glyph_width - bounding_box.xmin == official width - LSB
                RIGHT: bounding_box.width + bounding_box.xmin   == official width - RSB
                BOTH:  bounding_box.width                       == official width - LSB - RSB
        """
        if align is None:
            return self._width

        bounding_box = self.bounding_box()
        if align == ave.common.Align.LEFT:
            return self._width - bounding_box.xmin
        elif align == ave.common.Align.RIGHT:
            return bounding_box.xmin + bounding_box.width
        elif align == ave.common.Align.BOTH:
            return bounding_box.width
        else:
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

    @property
    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of a glyph (sign varies +/-).
        """
        return self.bounding_box().xmin

    @property
    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of a glyph (sign varies +/-).
        """
        return self._width - self.bounding_box().xmax

    def bounding_box(self) -> AvBox:
        """
        Returns bounding box (tightest box around Glyph)
        Coordinates are relative to baseline-origin (0,0) with orientation left-to-right, bottom-to-top
        Uses dimensions in unitsPerEm.
        """
        # Delegate entirely to AvPath's bounding box implementation
        return self._path.bounding_box()

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
        if self._character != other.character:
            return False

        # Width with numerical tolerance
        if not math.isclose(self._width, other.width(), rel_tol=rtol, abs_tol=atol):
            return False

        # Path with hierarchical comparison
        if not self._path.approx_equal(other.path, rtol, atol):
            return False

        return True

    def revise_direction(self) -> AvGlyph:
        """Normalize contour direction to TrueType/OpenType winding rules.

        TrueType/OpenType glyph outlines use contour winding to distinguish
        filled regions vs holes. This function rewrites each contour so
        that the glyph complies with the following standard directions:

        - Additive polygons (filled areas, outer contours): counter-clockwise (CCW)
        - Subtractive polygons (cut-out areas, inner contours/holes): clockwise (CW)

        Algorithm:
        Step 1: Split glyph into individual contours
        Step 2-3: Process each contour - check if closed, polygonize for area computation
        Step 4: Classify contours as additive vs subtractive using strict geometric nesting:
                - Check if a contour's bounding box is fully contained within another contour
                - Only if bounding boxes are nested, verify point containment
                - Non-nested contours are always additive (CCW)
                - Nested contours are subtractive (CW)
        Step 5: Enforce required direction - reverse contours that don't match winding rules
        Step 6: Reassemble contours into new AvPath

        Returns:
            AvGlyph: New glyph with corrected segment directions
        """
        # Step 1: Split glyph into individual contours
        contours: List[AvSinglePath] = self.path.split_into_single_paths()

        # Prepare lists for processed contours and their polygonized versions
        processed_contours: List[AvSinglePath] = []
        polygonized_contours: List[Optional[AvSinglePolygonPath]] = []  # For containment testing

        # Steps 2-3: Process each contour
        for contour in contours:
            # Check if contour is closed (ends with 'Z')
            is_closed = contour.commands and contour.commands[-1] == "Z"

            if not is_closed:
                # Edge case: leave open contours untouched
                processed_contours.append(contour)
                polygonized_contours.append(None)
                continue

            # Create a closed path to access polygonization utilities
            closed_path: AvClosedSinglePath = AvPath(
                contour.points.copy(), list(contour.commands), CLOSED_SINGLE_PATH_CONSTRAINTS
            )

            # Polygonize the contour for robust winding computation
            polygonized: AvSinglePolygonPath = closed_path.polygonized_path()

            # Edge case: skip degenerate contours with near-zero area
            if abs(polygonized.area) < 1e-10:
                processed_contours.append(contour)
                polygonized_contours.append(None)
                continue

            processed_contours.append(contour)
            polygonized_contours.append(polygonized)

        # Step 4: Simple classification - only check for actual geometric nesting
        contour_classes = []

        for i, (contour, polygonized) in enumerate(zip(processed_contours, polygonized_contours)):
            if polygonized is None:
                # Open contour - not classified
                contour_classes.append(None)
                continue

            # Get current area and test point
            current_area = abs(polygonized.area)
            test_point = polygonized.representative_point()

            # Check if this contour is geometrically nested inside another contour
            # Use strict test: bounding box must be fully contained AND area must be smaller
            is_nested = False
            current_bbox = polygonized.bounding_box()

            for j, other_polygonized in enumerate(polygonized_contours):
                if j != i and other_polygonized is not None:
                    other_area = abs(other_polygonized.area)
                    other_bbox = other_polygonized.bounding_box()

                    # Strict test: current bbox must be fully inside other bbox
                    bbox_fully_contained = (
                        current_bbox.xmin >= other_bbox.xmin
                        and current_bbox.xmax <= other_bbox.xmax
                        and current_bbox.ymin >= other_bbox.ymin
                        and current_bbox.ymax <= other_bbox.ymax
                    )

                    # Must also be smaller to be considered nested
                    if bbox_fully_contained and other_area > current_area:
                        if other_polygonized.contains_point(test_point):
                            is_nested = True
                            break

            # Classification: non-nested = additive (CCW), nested = subtractive (CW)
            is_additive = not is_nested
            contour_classes.append(is_additive)

        # Step 5: Enforce required direction
        final_contours: List[AvSinglePath] = []

        for contour, polygonized, is_additive in zip(processed_contours, polygonized_contours, contour_classes):
            if polygonized is None or is_additive is None:
                # Open contour - leave as is
                final_contours.append(contour)
                continue

            # Get current winding direction
            is_ccw = polygonized.is_ccw

            # Check if direction needs correction
            needs_reversal = False
            if is_additive and not is_ccw:
                # Additive should be CCW
                needs_reversal = True
            elif not is_additive and is_ccw:
                # Subtractive should be CW
                needs_reversal = True

            if needs_reversal:
                # Reverse the contour while preserving geometry
                reversed_contour = contour.reverse()
                final_contours.append(reversed_contour)
            else:
                final_contours.append(contour)

        # Step 6: Reassemble contours into new AvPath
        if final_contours:
            new_path = AvPath.join_paths(*final_contours)
        else:
            new_path = AvPath()

        # Return new AvGlyph with corrected directions
        return AvGlyph(character=self.character, width=self.width(), path=new_path)

    def validate(self) -> None:
        """
        Validate the glyph.
        Checks done:
            Step 1: Check correct direction of segments of path (additive: CCW, subtractive: CW) by
                calling revise_direction() and validate by comparing its result with self
            Step 2: Check width to be in range 0 < width < 10*bounding_box.width
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
        WIDTH_RANGE_MULTIPLIER = 10.0  # pylint: disable=invalid-name
        COORDINATE_RANGE_MULTIPLIER = 10.0  # pylint: disable=invalid-name
        UNITS_PER_EM_DEFAULT = 2048.0  # pylint: disable=invalid-name
        DUPLICATE_POINT_RTOL = 1e-9  # pylint: disable=invalid-name
        DUPLICATE_POINT_ATOL = 1e-9  # pylint: disable=invalid-name

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
            raise ValueError("Glyph path segments have incorrect winding directions")

        # Step 2: Check width range
        bbox = self.bounding_box()
        # Special case for space character: no visual bounding box but has glyph width
        if self._character == " " or bbox.width == 0:
            # For space character, only check that width is positive
            if self._width <= 0:
                raise ValueError(f"Space character width must be positive, got {self._width}")
        else:
            # For normal characters, check width is within reasonable range
            if not 0 < self._width < WIDTH_RANGE_MULTIPLIER * bbox.width:
                raise ValueError(
                    f"Glyph width {self._width} is not in valid range (0, {WIDTH_RANGE_MULTIPLIER * bbox.width})"
                )

        # Step 3: Validate path structure using existing path validation
        # Use closed path constraints to ensure all segments are closed

        # Check that all segments are closed
        segments = PathSplitter.split_commands_into_segments(self._path.commands)
        for seg_idx, (seg_cmds, _) in enumerate(segments):
            if not seg_cmds:
                raise ValueError(f"Segment {seg_idx} is empty")
            if seg_cmds[0] != "M":
                raise ValueError(f"Segment {seg_idx} must start with 'M' command")
            if seg_cmds[-1] != "Z":
                raise ValueError(f"Segment {seg_idx} must end with 'Z' command for closed path")

        # Step 4: Check coordinates for NaN and infinity
        points = self._path.points
        if not np.all(np.isfinite(points)):
            invalid_coords = np.where(~np.isfinite(points))
            raise ValueError(f"Glyph contains invalid coordinates (NaN or infinity) at indices: {invalid_coords}")

        # Step 5: Check for reasonable coordinate ranges
        min_coord = -COORDINATE_RANGE_MULTIPLIER * UNITS_PER_EM_DEFAULT
        max_coord = COORDINATE_RANGE_MULTIPLIER * UNITS_PER_EM_DEFAULT

        if np.any(points[:, 0] < min_coord) or np.any(points[:, 0] > max_coord):
            raise ValueError(f"Glyph x-coordinates exceed reasonable range [{min_coord}, {max_coord}]")
        if np.any(points[:, 1] < min_coord) or np.any(points[:, 1] > max_coord):
            raise ValueError(f"Glyph y-coordinates exceed reasonable range [{min_coord}, {max_coord}]")

        # Step 6: Check for reasonable bounding box
        if bbox.xmin < min_coord or bbox.xmax > max_coord or bbox.ymin < min_coord or bbox.ymax > max_coord:
            raise ValueError(f"Glyph bounding box exceeds reasonable range [{min_coord}, {max_coord}]")

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
                raise ValueError(f"Glyph contains duplicate consecutive points at indices: {duplicate_indices}")

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
    Abstract base class for glyph factories.
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

    def get_font_properties(self) -> Optional["AvFontProperties"]:
        """
        Return font properties if available, None otherwise.

        Default implementation returns None. Child classes should override
        this method if they can provide font properties, or forward the
        request to their source factory.

        Returns:
            Optional[AvFontProperties]: Font properties if available
        """
        return None


###############################################################################
# AvGlyphCachedFactory
###############################################################################


@dataclass
class AvGlyphCachedFactory(AvGlyphFactory):
    """Glyph factory backed by an in-memory glyph dictionary with optional fallback.

    The factory first tries to return a glyph from its internal ``_glyphs`` cache.
    If the glyph is not present and a ``_source_factory`` is configured, it will
    delegate creation to that factory, cache the result, and return it.
    """

    _glyphs: Dict[str, AvGlyph] = field(default_factory=dict)
    _source_factory: Optional[AvGlyphFactory] = None

    def __init__(
        self,
        glyphs: Optional[Dict[str, AvGlyph]] = None,
        source_factory: Optional[AvGlyphFactory] = None,
    ) -> None:
        """Initialize the factory with an optional glyph cache and source factory."""
        if glyphs is None:
            glyphs = {}
        self._glyphs = glyphs
        self._source_factory = source_factory

    @property
    def glyphs(self) -> Dict[str, AvGlyph]:
        """Return the internal glyph cache mapping characters to glyph instances."""
        return self._glyphs

    def to_cache_dict(self) -> dict:
        """
        Create a dictionary representation of the cached glyphs.

        Returns:
            dict: Dictionary with "glyphs" key containing glyph data.
        """
        glyphs_dict: Dict[str, dict] = {character: glyph.to_dict() for character, glyph in self._glyphs.items()}
        return {
            "glyphs": glyphs_dict,
        }

    @classmethod
    def from_cache_dict(cls, data: dict) -> "AvGlyphCachedFactory":
        """
        Create a cached factory from cache data dictionary.

        Args:
            data: Dictionary with "glyphs" key containing a dictionary of glyph data.

        Returns:
            AvGlyphCachedFactory: Factory with loaded glyphs
        """
        glyph_entries = data.get("glyphs", {})
        glyphs: Dict[str, AvGlyph] = {}

        for character, glyph_data in glyph_entries.items():
            try:
                glyphs[character] = AvGlyph.from_dict(glyph_data)
            except (ValueError, KeyError, TypeError) as e:
                # Expected errors when glyph data is malformed
                print(f"Warning: Failed to load glyph for '{character}': {e}")
            except Exception as e:
                # Unexpected error - re-raise with context
                raise RuntimeError(f"Unexpected error loading glyph '{character}': {e}") from e

        return cls(glyphs=glyphs, source_factory=None)

    @property
    def source_factory(self) -> Optional[AvGlyphFactory]:
        """Return the optional backing glyph factory used as a cache miss source."""
        return self._source_factory

    @source_factory.setter
    def source_factory(self, source_factory: Optional[AvGlyphFactory]) -> None:
        self._source_factory = source_factory

    def get_glyph(self, character: str) -> AvGlyph:
        """
        Retrieve a glyph from the cache or source factory.

        First attempts to return a glyph from its internal ``_glyphs`` cache.
        If the glyph is not present and a ``_source_factory`` is configured, it will
        delegate creation to that factory, cache the result, and return it.

        Args:
            character (str): The character to retrieve.

        Returns:
            AvGlyph: The cached glyph instance.

        Raises:
            KeyError: If glyph is not found in cache or source factory.
        """
        if character in self._glyphs:
            return self._glyphs[character]

        if self._source_factory is None:
            raise KeyError(f"Glyph for character {character!r} not found and no source_factory provided.")

        glyph = self._source_factory.get_glyph(character)
        self._glyphs[character] = glyph
        return glyph

    def get_font_properties(self) -> Optional["AvFontProperties"]:
        """
        Return font properties from source factory if available.

        Forwards the request to the source factory since the cached factory
        doesn't have font properties of its own.

        Returns:
            Optional[AvFontProperties]: Font properties from source factory
        """
        if self._source_factory is not None:
            return self._source_factory.get_font_properties()
        return None


###############################################################################
# AvGlyphFromTTFontFactory
###############################################################################


@dataclass
class AvGlyphFromTTFontFactory(AvGlyphFactory):
    """Factory class for creating glyph instances."""

    _ttfont: TTFont

    def __init__(self, ttfont: TTFont) -> None:
        """
        Initializes the glyph factory.
        """
        self._ttfont = ttfont

    @property
    def ttfont(self) -> TTFont:
        """
        Returns the TTFont instance associated with this glyph factory.
        """
        return self._ttfont

    def get_glyph(self, character: str) -> AvGlyph:
        return AvGlyph.from_ttfont_character(self._ttfont, character)

    def get_font_properties(self) -> Optional["AvFontProperties"]:
        """
        Return font properties extracted from the TTFont.

        Returns:
            Optional[AvFontProperties]: Font properties from TTFont
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

    _source_factory: AvGlyphFactory
    _polygonize_steps: int

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
            width=original_glyph.width(),
            path=polygonized_path,
        )

    def get_font_properties(self) -> Optional["AvFontProperties"]:
        """
        Return font properties from the source factory.

        Forwards the request to the source factory since polygonization
        doesn't affect font properties.

        Returns:
            Optional[AvFontProperties]: Font properties from source factory
        """
        return self._source_factory.get_font_properties()


###############################################################################
# Letter
###############################################################################


@dataclass
class AvLetter:
    """
    A Letter is a Glyph which is scaled to real dimensions with a position and alignment.
    """

    _glyph: AvGlyph
    _scale: float  # = font_size / units_per_em
    _xpos: float  # left-to-right
    _ypos: float  # bottom-to-top
    _align: Optional[ave.common.Align] = None  # LEFT, RIGHT, BOTH. Defaults to None.

    def __init__(
        self,
        glyph: AvGlyph,
        scale: float,
        xpos: float = 0.0,
        ypos: float = 0.0,
        align: Optional[ave.common.Align] = None,
    ) -> None:
        self._glyph = glyph
        self._scale = scale
        self._xpos = xpos
        self._ypos = ypos
        self._align = align

    @classmethod
    def from_font_size_units_per_em(
        cls,
        glyph: AvGlyph,
        font_size: float,
        units_per_em: float,
        xpos: float = 0.0,
        ypos: float = 0.0,
        align: Optional[ave.common.Align] = None,
    ) -> AvLetter:
        """
        Factory method to create an AvLetter from font_size and units_per_em.
        """

        return cls(glyph, font_size / units_per_em, xpos, ypos, align)

    @property
    def xpos(self) -> float:
        """The x position of the letter in real dimensions."""
        return self._xpos

    @xpos.setter
    def xpos(self, xpos: float) -> None:
        """Sets the x position of the letter in real dimensions."""
        self._xpos = xpos

    @property
    def ypos(self) -> float:
        """The y position of the letter in real dimensions."""
        return self._ypos

    @ypos.setter
    def ypos(self, ypos: float) -> None:
        """Sets the y position of the letter in real dimensions."""
        self._ypos = ypos

    @property
    def scale(self) -> float:
        """Returns the scale factor for the letter which is used to transform the glyph to real dimensions."""
        return self._scale

    @property
    def align(self) -> Optional[ave.common.Align]:
        """The alignment of the letter; None, LEFT, RIGHT, BOTH."""
        return self._align

    @property
    def trafo(self) -> List[float]:
        """
        Returns the affine transformation matrix for the letter to transform the glyph to real dimensions.
        Returns: [scale, 0, 0, scale, xpos, ypos] or [scale, 0, 0, scale, xpos-lsb, ypos] if alignment is LEFT or BOTH.
        """
        if self.align == ave.common.Align.LEFT or self.align == ave.common.Align.BOTH:
            lsb_scaled = self.scale * self._glyph.left_side_bearing
            return [self.scale, 0, 0, self.scale, self.xpos - lsb_scaled, self.ypos]
        return [self.scale, 0, 0, self.scale, self.xpos, self.ypos]

    @property
    def width(self) -> float:
        """
        Returns the width calculated considering the alignment.
        """
        return self.scale * self._glyph.width(self.align)

    @property
    def height(self) -> float:
        """
        The height of the Letter, i.e. the height of the bounding box.
        """
        return self.scale * self._glyph.height

    @property
    def ascender(self) -> float:
        """
        The maximum distance above the baseline, i.e. the highest y-coordinate of a Letter (positive value).
        """
        return self.scale * self._glyph.ascender

    @property
    def descender(self) -> float:
        """
        The maximum distance below the baseline, i.e. the lowest y-coordinate of a Letter (negative value).
        """
        return self.scale * self._glyph.descender

    @property
    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of the Letter taking alignment into account.
        """
        if self.align == ave.common.Align.LEFT or self.align == ave.common.Align.BOTH:
            return 0
        return self.scale * self._glyph.left_side_bearing

    @property
    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of the Letter taking alignment into account.
        """
        if self.align == ave.common.Align.RIGHT or self.align == ave.common.Align.BOTH:
            return 0
        return self.scale * self._glyph.right_side_bearing

    def bounding_box(self) -> AvBox:
        """
        Returns bounding box (tightest box around Letter) in real dimensions.
        Coordinates are relative to baseline-origin (0,0) with orientation left-to-right, bottom-to-top
        Returns:
            AvBox: The bounding box of the letter.
        """
        return self._glyph.bounding_box().transform_affine(self.trafo)

    def svg_path_string(self) -> str:
        """
        Returns the SVG path representation of the letter in real dimensions.
        The SVG path is a string that defines the outline of the letter using
        SVG path commands. This path can be used to render the letter as a
        vector graphic.
        Returns:
            str: The SVG path string representing the letter.
        """
        points = self._glyph.path.points
        commands = self._glyph.path.commands
        scale, _, _, _, translate_x, translate_y = self.trafo
        return AvLetter._svg_path_string(points, commands, scale, translate_x, translate_y)

    def svg_path_string_debug_polyline(self, stroke_width: float = 1.0) -> str:
        """
        Returns a debug SVG path representation of the letter using only polylines.
        This method converts curves (Q, C) to straight lines between control points,
        making it useful for debugging the path structure.

        Supported commands:
            M (move-to), L (line-to), Z (close-path)
            Q (quadratic bezier) -> converted to L commands
            C (cubic bezier) -> converted to L commands

        Args:
            stroke_width: The stroke width used to determine marker sizes.

        Returns:
            str: The debug SVG path string using only polylines with markers at each point.
                Markers include:
                - Squares: Regular points (L commands)
                - Circles: Control points (intermediate points in Q and C commands)
                - Triangles (pointing right): M command points (segment starts)
                - Triangles (pointing left): Points before Z commands (segment ends)
                Note: Triangles are drawn in addition to the base markers (squares/circles).
        """
        points = self._glyph.path.points
        commands = self._glyph.path.commands
        scale, _, _, _, translate_x, translate_y = self.trafo
        return AvLetter._svg_path_string_debug_polyline(points, commands, scale, translate_x, translate_y, stroke_width)

    @classmethod
    def _svg_path_string(
        cls,
        points: NDArray[np.float64],
        commands: List[AvGlyphCmds],
        scale: float = 1.0,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
    ) -> str:
        """
        Returns the SVG path representation (absolute coordinates) of the glyph.
        The SVG path is a string that defines the outline of the glyph using
        SVG path commands. This path can be used to render the glyph as a
        vector graphic.

        Supported commands:
            M (move-to), L (line-to),
            C (cubic bezier), Q (quadratic bezier),
            Z (close-path).

        Args:
            points (NDArray[np.float64]): The array of points defining the outline of the glyph.
            commands (List[AvGlyphCmds]): The list of commands defining the outline of the glyph.
            scale (float): The scale factor to apply to the points before generating the SVG path string.
            translate_x (float): X-coordinate translation before generating the SVG path string.
            translate_y (float): Y-coordinate translation before generating the SVG path string.

        Returns:
            str: The SVG path string (absolute coordinates) representing the glyph.
                    Returns "M 0 0" if there are no points.
        """
        # Apply scale and translation to the points, make points to be 2 dimensions (x, y)
        points_transformed = points[:, :2] * scale + (translate_x, translate_y)

        parts: List[str] = []
        p_idx = 0
        for cmd in commands:
            if cmd == "M" or cmd == "L":
                # Move-to or Line-to: one point (x,y)
                if p_idx >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x, y = points_transformed[p_idx]
                parts.append(f"{cmd} {x:g} {y:g}")
                p_idx += 1
            elif cmd == "Q":
                # Quadratic bezier: control point + end point (2 points total)
                if p_idx + 1 >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x1, y1 = points_transformed[p_idx]
                x2, y2 = points_transformed[p_idx + 1]
                parts.append(f"{cmd} {x1:g} {y1:g} {x2:g} {y2:g}")
                p_idx += 2
            elif cmd == "C":
                # Cubic bezier: control1 + control2 + end point (3 points total)
                if p_idx + 2 >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x1, y1 = points_transformed[p_idx]
                x2, y2 = points_transformed[p_idx + 1]
                x3, y3 = points_transformed[p_idx + 2]
                parts.append(f"{cmd} {x1:g} {y1:g} {x2:g} {y2:g} {x3:g} {y3:g}")
                p_idx += 3
            elif cmd == "Z":
                # Close-path: no coordinates
                parts.append("Z")
            else:
                # Unsupported command (should not occur from AvPointCommandPen)
                raise ValueError(f"Unsupported SVG command in AvGlyph: {cmd}")

        # Return the composed absolute-path string or "M 0 0" string if parts is empty
        return " ".join(parts) if parts else "M 0 0"

    @classmethod
    def _svg_path_string_debug_polyline(
        cls,
        points: NDArray[np.float64],
        commands: List[AvGlyphCmds],
        scale: float = 1.0,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
        stroke_width: float = 1.0,
    ) -> str:
        """
        Returns a debug SVG path representation using only polylines.
        This method converts curves (Q, C) to straight lines between control points.

        Supported commands:
            M (move-to), L (line-to), Z (close-path)
            Q (quadratic bezier) -> converted to L commands
            C (cubic bezier) -> converted to L commands

        Args:
            points (NDArray[np.float64]): The array of points defining the outline of the glyph.
            commands (List[AvGlyphCmds]): The list of commands defining the outline of the glyph.
            scale (float): The scale factor to apply to the points before generating the SVG path string.
            translate_x (float): X-coordinate translation before generating the SVG path string.
            translate_y (float): Y-coordinate translation before generating the SVG path string.
            stroke_width (float): The stroke width used to determine square marker size.

        Returns:
            str: The debug SVG path string using only polylines with markers.
                Markers include:
                - Squares: Regular points (L commands)
                - Circles: Control points (intermediate points in Q and C commands)
                - Triangles (pointing right): M command points (segment starts)
                - Triangles (pointing left): Points before Z commands (segment ends)
                Note: Triangles are drawn in addition to the base markers (squares/circles).
                    Returns "M 0 0" if there are no points.
        """
        # Apply scale and translation to the points, make points to be 2 dimensions (x, y)
        points_transformed = points[:, :2] * scale + (translate_x, translate_y)

        parts: List[str] = []
        p_idx = 0

        for cmd in commands:
            if cmd == "M":
                # Move-to: one point (x,y), start new segment
                if p_idx >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x, y = points_transformed[p_idx]
                parts.append(f"M {x:g} {y:g}")
                p_idx += 1
            elif cmd == "L":
                # Line-to: one point (x,y)
                if p_idx >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x, y = points_transformed[p_idx]
                parts.append(f"L {x:g} {y:g}")
                p_idx += 1
            elif cmd == "Q":
                # Quadratic bezier: control point + end point -> convert to 2 L commands
                if p_idx + 1 >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x1, y1 = points_transformed[p_idx]  # Control point
                x2, y2 = points_transformed[p_idx + 1]  # End point
                parts.append(f"L {x1:g} {y1:g}")  # Line to control point
                parts.append(f"L {x2:g} {y2:g}")  # Line to end point
                p_idx += 2
            elif cmd == "C":
                # Cubic bezier: control1 + control2 + end point -> convert to 3 L commands
                if p_idx + 2 >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x1, y1 = points_transformed[p_idx]  # Control point 1
                x2, y2 = points_transformed[p_idx + 1]  # Control point 2
                x3, y3 = points_transformed[p_idx + 2]  # End point
                parts.append(f"L {x1:g} {y1:g}")  # Line to control point 1
                parts.append(f"L {x2:g} {y2:g}")  # Line to control point 2
                parts.append(f"L {x3:g} {y3:g}")  # Line to end point
                p_idx += 3
            elif cmd == "Z":
                # Close-path: draw line to start of current segment
                parts.append("Z")
                # No need to track segment_start_point anymore as segment is closed
            else:
                # Unsupported command (should not occur from AvPointCommandPen)
                raise ValueError(f"Unsupported SVG command in AvGlyph: {cmd}")

        # Add markers at each point
        square_size = stroke_width * 2
        half_size = square_size / 2
        circle_radius = square_size / 2

        # Track which points are control points and special markers
        p_idx = 0
        is_control_point = [False] * len(points_transformed)
        is_m_point = [False] * len(points_transformed)  # M command points
        is_before_z_point = [False] * len(points_transformed)  # Points before Z commands

        # First pass: identify control points and special markers
        for i, cmd in enumerate(commands):
            if cmd == "M":
                # Move-to: mark this point
                if p_idx < len(is_m_point):
                    is_m_point[p_idx] = True
                p_idx += 1
            elif cmd == "L":
                # Line-to: regular point
                p_idx += 1
            elif cmd == "Q":
                # Quadratic bezier: control point + end point (2 points total)
                # First point is control point, second is end point
                if p_idx < len(is_control_point):
                    is_control_point[p_idx] = True  # Control point
                p_idx += 2
            elif cmd == "C":
                # Cubic bezier: control1 + control2 + end point (3 points total)
                # First two points are control points, third is end point
                if p_idx < len(is_control_point):
                    is_control_point[p_idx] = True  # Control point 1
                if p_idx + 1 < len(is_control_point):
                    is_control_point[p_idx + 1] = True  # Control point 2
                p_idx += 3
            elif cmd == "Z":
                # Close-path: mark the previous point as before-Z
                # Find the last point before this Z command
                if i > 0:
                    # Count points used before this Z to find the last point index
                    temp_p_idx = 0
                    for j in range(i):
                        prev_cmd = commands[j]
                        if prev_cmd == "M" or prev_cmd == "L":
                            temp_p_idx += 1
                        elif prev_cmd == "Q":
                            temp_p_idx += 2
                        elif prev_cmd == "C":
                            temp_p_idx += 3
                        # Z commands don't use points

                    if temp_p_idx > 0:  # There is a point before this Z
                        last_point_idx = temp_p_idx - 1
                        if last_point_idx < len(is_before_z_point):
                            is_before_z_point[last_point_idx] = True
            else:
                # Unsupported command (should not occur from AvPointCommandPen)
                raise ValueError(f"Unsupported SVG command in AvGlyph: {cmd}")

        # Add markers for all transformed points
        for i, (x, y) in enumerate(points_transformed):
            # Always add the base marker (square or circle)
            if is_control_point[i]:
                # Control point: circle
                parts.append(f"M {x - circle_radius:g} {y:g}")
                parts.append(f"A {circle_radius:g} {circle_radius:g} 0 1 0 {x + circle_radius:g} {y:g}")
                parts.append(f"A {circle_radius:g} {circle_radius:g} 0 1 0 {x - circle_radius:g} {y:g}")
            else:
                # Regular point: square
                square_x1 = x - half_size
                square_y1 = y - half_size
                square_x2 = x + half_size
                square_y2 = y + half_size

                parts.append(f"M {square_x1:g} {square_y1:g}")
                parts.append(f"L {square_x2:g} {square_y1:g}")
                parts.append(f"L {square_x2:g} {square_y2:g}")
                parts.append(f"L {square_x1:g} {square_y2:g}")
                parts.append("Z")

            # Add additional triangle markers for M points and before-Z points
            if is_m_point[i]:
                # M command point: equilateral triangle with left side vertical (pointing right)
                triangle_size = stroke_width * 2
                height = triangle_size * (3**0.5) / 2  # Height of equilateral triangle

                # Triangle with left side vertical (pointing right)
                # Left side is vertical, so vertices are:
                # Top: (x, y + height/2)
                # Bottom: (x, y - height/2)
                # Right: (x + triangle_size, y)
                parts.append(f"M {x:g} {y + height/2:g}")
                parts.append(f"L {x + triangle_size:g} {y:g}")
                parts.append(f"L {x:g} {y - height/2:g}")
                parts.append("Z")

            if is_before_z_point[i]:
                # Point before Z command: equilateral triangle with right side vertical (pointing left)
                triangle_size = stroke_width * 2
                height = triangle_size * (3**0.5) / 2  # Height of equilateral triangle

                # Triangle with right side vertical (pointing left)
                # Right side is vertical, so vertices are:
                # Top: (x, y + height/2)
                # Bottom: (x, y - height/2)
                # Left: (x - triangle_size, y)
                parts.append(f"M {x:g} {y + height/2:g}")
                parts.append(f"L {x - triangle_size:g} {y:g}")
                parts.append(f"L {x:g} {y - height/2:g}")
                parts.append("Z")

        # Return the composed absolute-path string or "M 0 0" string if parts is empty
        return " ".join(parts) if parts else "M 0 0"


###############################################################################
# Main
###############################################################################


def main():
    """Main"""


if __name__ == "__main__":
    main()
