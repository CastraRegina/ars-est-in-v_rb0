"""Font glyph handling and typography utilities for OpenType and SVG fonts."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from fontTools.ttLib import TTFont
from numpy.typing import NDArray

import ave.common
from ave.common import AvGlyphCmds
from ave.fonttools import AvGlyphPtsCmdsPen
from ave.geom import AvBox
from ave.path import (
    CLOSED_SINGLE_PATH_CONSTRAINTS,
    AvClosedSinglePath,
    AvPath,
    AvSinglePath,
    AvSinglePolygonPath,
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
        bounding_box = self.bounding_box()
        if align is None:
            return self._width
        elif align == ave.common.Align.LEFT:
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
        Step 4: Classify contours as additive vs subtractive using geometric nesting depth:
                - Count how many other contours contain each contour's interior point
                - Even depth = additive, Odd depth = subtractive
        Step 5: Enforce required direction - reverse contours that don't match winding rules
        Step 6: Reassemble contours into new AvPath

        Returns:
            AvGlyph: New glyph with corrected segement directions
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

        # Step 4: Classify contours as additive vs subtractive using nesting depth
        contour_classes: List[Optional[bool]] = []  # True for additive, False for subtractive

        for i, (contour, polygonized) in enumerate(zip(processed_contours, polygonized_contours)):
            if polygonized is None:
                # Open contour - not classified
                contour_classes.append(None)
                continue

            # Get a test point inside the contour
            test_point: tuple[float, float] = polygonized.representative_point()
            current_area = polygonized.area

            # Count how many other closed contours contain this point
            containment_depth = 0
            for j, other_polygonized in enumerate(polygonized_contours):
                if j != i and other_polygonized is not None:
                    if other_polygonized.area > current_area and other_polygonized.contains_point(test_point):
                        containment_depth += 1

            # Even depth => additive, odd depth => subtractive
            is_additive = containment_depth % 2 == 0
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
        return AvGlyph(self.character, self.width(), new_path)


###############################################################################
# Glyph Factory
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

    @property
    def source_factory(self) -> Optional[AvGlyphFactory]:
        """Return the optional backing glyph factory used as a cache miss source."""
        return self._source_factory

    @source_factory.setter
    def source_factory(self, source_factory: Optional[AvGlyphFactory]) -> None:
        self._source_factory = source_factory

    def get_glyph(self, character: str) -> AvGlyph:
        if character in self._glyphs:
            return self._glyphs[character]

        if self._source_factory is None:
            raise KeyError(f"Glyph for character {character!r} not found and no source_factory provided.")

        glyph = self._source_factory.get_glyph(character)
        self._glyphs[character] = glyph
        return glyph


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


###############################################################################
# Main
###############################################################################


def main():
    """Main"""


if __name__ == "__main__":
    main()
