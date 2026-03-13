"""Multi-weight letter implementation using AvSingleGlyphLetter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ave.common import AlignX
from ave.geom import AvBox
from ave.glyph import AvGlyph
from ave.glyph_factory import AvGlyphFactory
from ave.path import AvPath, AvSinglePolygonPath

###############################################################################
# AvLetter
###############################################################################


class AvLetter(ABC):
    """
    Base class for letter representations.

    A Letter is a visual representation of one or more glyphs positioned
    in real-world coordinates with scaling, alignment, and positioning.

    Following properties / methods need to be implemented by child classes:
        advance_width(self)
        bounding_box(self)
        letter_box(self)
        polygonize_path(self, steps: int = 50)
        svg_path_string(self)
        centroid(self)
        exterior(self, steps: int = 20)
        exterior_path(self)
        exterior_path_left_silhouette(self)
        exterior_path_right_silhouette(self)
    """

    _scale: float
    _xpos: float
    _ypos: float
    _left_letter: Optional[AvLetter] = None
    _right_letter: Optional[AvLetter] = None

    def __init__(
        self,
        scale: float,
        xpos: float,
        ypos: float,
        left_letter: Optional[AvLetter] = None,
        right_letter: Optional[AvLetter] = None,
    ) -> None:
        """Initialize AvLetter with position, scale and neighbors.

        Args:
            scale: Scale factor from glyph coordinates to real dimensions.
            xpos: X position in real dimensions (left-to-right).
            ypos: Y position in real dimensions (bottom-to-top).
            left_letter: Left neighbor letter.
            right_letter: Right neighbor letter.
        """
        self._scale = scale
        self._xpos = xpos
        self._ypos = ypos
        self._left_letter = left_letter
        self._right_letter = right_letter

    @property
    def scale(self) -> float:
        """Returns the scale factor that transforms glyph units (unitsPerEm) to real-world dimensions."""
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        """Sets the scale factor that transforms glyph units (unitsPerEm) to real-world dimensions."""
        self._scale = scale

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
    def left_letter(self) -> Optional["AvLetter"]:
        """The left neighbor letter."""
        return self._left_letter

    @left_letter.setter
    def left_letter(self, left_letter: Optional["AvLetter"]) -> None:
        """Sets the left neighbor letter."""
        self._left_letter = left_letter

    @property
    def right_letter(self) -> Optional["AvLetter"]:
        """The right neighbor letter."""
        return self._right_letter

    @right_letter.setter
    def right_letter(self, right_letter: Optional["AvLetter"]) -> None:
        """Sets the right neighbor letter."""
        self._right_letter = right_letter

    @property
    def ascender(self) -> float:
        """Maximum distance above baseline from letter's bounding box.

        Calculated as bounding_box.ymax.
        """
        return self.bounding_box.ymax

    @property
    def descender(self) -> float:
        """Maximum distance below baseline from letter's bounding box.

        Calculated as bounding_box.ymin.
        """
        return self.bounding_box.ymin

    @property
    def height(self) -> float:
        """Height of letter from letter's bounding box.

        Calculated as bounding_box.height.
        """
        return self.bounding_box.height

    @property
    def left_side_bearing(self) -> float:
        """Horizontal space on left side of letter.

        Usually calculated as bounding_box.xmin - letter_box.xmin.
        Positive when glyph is right of origin, negative when left of origin.
        """
        return self.bounding_box.xmin - self.letter_box.xmin

    @property
    def right_side_bearing(self) -> float:
        """Horizontal space on right side of letter.

        Usually calculated as advance_width - (bounding_box.xmax - letter_box.xmin).
        Positive when bounding box is inside advance box, negative when extends beyond.
        """
        return self.advance_width - (self.bounding_box.xmax - self.letter_box.xmin)

    def width(self, align: Optional[AlignX] = None) -> float:
        """Returns the width calculated considering the alignment.

        For None: returns advance_width.
        For LEFT: returns advance_width - left_side_bearing.
        For RIGHT: returns advance_width - right_side_bearing.
        For BOTH: returns bounding_box.width.
        """
        if align is None:
            return self.advance_width
        if align == AlignX.LEFT:
            return self.advance_width - self.left_side_bearing
        if align == AlignX.RIGHT:
            return self.advance_width - self.right_side_bearing
        if align == AlignX.BOTH:
            return self.bounding_box.width
        return self.advance_width

    ####################################################################################

    @property
    @abstractmethod
    def advance_width(self) -> float:
        """Returns the advance width.

        Usually calculated as scale * glyph.advance_width.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def bounding_box(self) -> AvBox:
        """Returns the bounding box around the outline.

        Usually calculated by transforming glyph's bounding_box with the letter's scale and position.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def letter_box(self) -> AvBox:
        """Returns the letter's advance box.

        Usually created by transforming glyph's glyph_box (AvBox(0, descender, advance_width, ascender))
        using the letter's scale and position.
        """
        raise NotImplementedError

    @abstractmethod
    def polygonize_path(self, steps: int = 50) -> Optional[AvPath]:
        """Return the polygonized outline in world coordinates.

        The returned ``AvPath`` is the polygonized letter outline with
        all coordinates already transformed to world space (scaled and
        translated).

        Args:
            steps: Number of segments for curve approximation (default: 50).

        Returns:
            Polygonized path in world coordinates, or ``None`` when the
            letter has no geometry (e.g. empty glyph path).
        """
        raise NotImplementedError

    @abstractmethod
    def svg_path_string(self) -> str:
        """Returns the SVG path representation of the letter."""
        raise NotImplementedError

    @property
    @abstractmethod
    def centroid(self) -> Tuple[float, float]:
        """Returns the centroid of the letter in real dimensions.

        The centroid is the geometric center of the letter's outline,
        calculated from the actual path geometry (not from the bounding box)
        and transformed to world coordinates.

        Returns:
            Tuple[float, float]: The (x, y) coordinates of the centroid.
        """
        raise NotImplementedError

    @abstractmethod
    def exterior(self, steps: int = 20) -> List[AvSinglePolygonPath]:
        """Get the exterior paths of the letter without holes.

        Returns the exterior (positive) polygons of the letter with all holes
        and cut-outs removed. The paths are transformed to the letter's
        coordinate system.

        Args:
            steps: Number of segments to use for curve approximation during polygonization

        Returns:
            List of AvSinglePolygonPath objects representing only positive polygons
            (exterior rings without any holes) in the letter's coordinate system
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def exterior_path(self) -> List[AvSinglePolygonPath]:
        """Convert glyph outline to one or more polygons without holes using fixed internal steps.

        Returns:
            List of transformed AvSinglePolygonPath objects representing only positive polygons
            (exterior rings without any holes) in the letter's coordinate system
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def exterior_path_left_silhouette(self) -> List[AvSinglePolygonPath]:
        """Get left orthographic silhouette of the letter's exterior polygons.

        Returns:
            List of AvSinglePolygonPath objects representing the left silhouette
            with right blocking edge at x = max_x in the letter's coordinate system
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def exterior_path_right_silhouette(self) -> List[AvSinglePolygonPath]:
        """Get right orthographic silhouette of the letter's exterior polygons.

        Returns:
            List of AvSinglePolygonPath objects representing the right silhouette
            with left blocking edge at x = min_x in the letter's coordinate system
        """
        raise NotImplementedError


###############################################################################
# AvSingleGlyphLetter
###############################################################################


class AvSingleGlyphLetter(AvLetter):
    """
    A Letter is a Glyph which is scaled to real dimensions with a position.
    """

    _glyph: AvGlyph

    def __init__(
        self,
        glyph: AvGlyph,
        scale: float = 1.0,  # scale from glyph-coordinates to real dimensions (font_size / units_per_em)
        xpos: float = 0.0,  # left-to-right, value in real dimensions
        ypos: float = 0.0,  # bottom-to-top, value in real dimensions
        left_letter: Optional[AvLetter] = None,
        right_letter: Optional[AvLetter] = None,
    ) -> None:
        super().__init__(scale, xpos, ypos, left_letter, right_letter)
        self._glyph = glyph

    @property
    def glyph(self) -> AvGlyph:
        """The glyph of the letter."""
        return self._glyph

    ####################################################################################

    @property
    def advance_width(self) -> float:
        """Returns the advance width of the letter in real dimensions."""
        return self.scale * self._glyph.advance_width

    @property
    def bounding_box(self) -> AvBox:
        """Returns the tightest bounding box around the outline in real dimensions.

        Transforms glyph's bounding_box with scale and position.
        Coordinates are relative to baseline-origin (0,0) with orientation
        left-to-right, bottom-to-top.

        Returns:
            AvBox: The bounding box in world coordinates.
        """
        return self._glyph.bounding_box.transform_scale_translate(self.scale, self.xpos, self.ypos)

    @property
    def letter_box(self) -> AvBox:
        """Returns the letter's advance box in real dimensions.

        Transforms glyph's glyph_box (AvBox(0, descender, advance_width, ascender))
        with scale and position. Unlike bounding_box, includes full advance width.

        Returns:
            AvBox: The transformed glyph advance box.
        """
        return self._glyph.glyph_box.transform_scale_translate(self.scale, self.xpos, self.ypos)

    def polygonize_path(self, steps: int = 50) -> Optional[AvPath]:
        """Return the polygonized outline in world coordinates.

        Args:
            steps: Number of segments for curve approximation (default: 50).

        Returns:
            Polygonized path in world coordinates, or ``None`` if the
            glyph path is empty.
        """
        if self._glyph.path.points.size == 0:
            return None
        return self._glyph.path.polygonize(steps).transformed_copy(self.scale, self.xpos, self.ypos)

    def svg_path_string(self) -> str:
        """
        Returns the SVG path representation of the letter in real dimensions.
        The SVG path is a string that defines the outline of the letter using
        SVG path commands. This path can be used to render the letter as a
        vector graphic.
        Returns:
            str: The SVG path string representing the letter.
        """
        return self._glyph.path.svg_path_string(self.scale, self.xpos, self.ypos)

    @property
    def centroid(self) -> Tuple[float, float]:
        """
        Returns the centroid of the letter in real dimensions.

        The centroid is the geometric center of the letter's outline,
        calculated from the actual path geometry (not from the bounding box)
        and transformed to world coordinates.

        Returns:
            Tuple[float, float]: The (x, y) coordinates of the centroid.
        """
        # Get the glyph's centroid and transform it using the letter's transformation
        glyph_centroid = self._glyph.centroid

        centroid_x = glyph_centroid[0] * self.scale + self.xpos
        centroid_y = glyph_centroid[1] * self.scale + self.ypos

        return (centroid_x, centroid_y)

    def exterior(self, steps: int = 20) -> List[AvSinglePolygonPath]:
        """Get the exterior paths of the letter without holes.

        This method calls the glyph's exterior method and applies the same
        scale and translation transformations as the letter.

        Args:
            steps: Number of segments to use for curve approximation during polygonization

        Returns:
            List of transformed AvSinglePolygonPath objects representing only positive polygons
            (exterior rings without any holes) in the letter's coordinate system
        """
        # Get exterior paths from the glyph
        exterior_paths = self._glyph.exterior(steps)

        # Apply the letter's transformation to each exterior path
        transformed_paths = []
        for path in exterior_paths:
            transformed_path = path.transformed_copy(self.scale, self.xpos, self.ypos)
            transformed_paths.append(transformed_path)

        return transformed_paths

    @property
    def exterior_path(self) -> List[AvSinglePolygonPath]:
        """Convert glyph outline to one or more polygons without holes using fixed internal steps.

        Uses the glyph's cached exterior_path and applies the same scale and translation
        transformations as the letter.

        Returns:
            List of transformed AvSinglePolygonPath objects representing only positive polygons
            (exterior rings without any holes) in the letter's coordinate system
        """
        # Get exterior paths from the glyph
        exterior_paths = self._glyph.exterior_path

        # Apply the letter's transformation to each exterior path
        transformed_paths = []
        for path in exterior_paths:
            transformed_path = path.transformed_copy(self.scale, self.xpos, self.ypos)
            transformed_paths.append(transformed_path)

        return transformed_paths

    @property
    def exterior_path_left_silhouette(self) -> List[AvSinglePolygonPath]:
        """Get left orthographic silhouette of the letter's exterior polygons.

        Uses the glyph's cached exterior_path_left_silhouette and applies the same
        scale and translation transformations as the letter.

        Returns:
            List of AvSinglePolygonPath objects representing the left silhouette
            with right blocking edge at x = max_x in the letter's coordinate system
        """
        # Get left silhouette paths from the glyph
        silhouette_paths = self._glyph.exterior_path_left_silhouette

        # Apply the letter's transformation to each silhouette path
        transformed_paths = []
        for path in silhouette_paths:
            transformed_path = path.transformed_copy(self.scale, self.xpos, self.ypos)
            transformed_paths.append(transformed_path)

        return transformed_paths

    @property
    def exterior_path_right_silhouette(self) -> List[AvSinglePolygonPath]:
        """Get right orthographic silhouette of the letter's exterior polygons.

        Uses the glyph's cached exterior_path_right_silhouette and applies the same
        scale and translation transformations as the letter.

        Returns:
            List of AvSinglePolygonPath objects representing the right silhouette
            with left blocking edge at x = min_x in the letter's coordinate system
        """
        # Get right silhouette paths from the glyph
        silhouette_paths = self._glyph.exterior_path_right_silhouette

        # Apply the letter's transformation to each silhouette path
        transformed_paths = []
        for path in silhouette_paths:
            transformed_path = path.transformed_copy(self.scale, self.xpos, self.ypos)
            transformed_paths.append(transformed_path)

        return transformed_paths

    ####################################################################################

    def svg_path_string_debug_polyline(self, stroke_width: float = 1.0) -> str:
        """
        Returns a debug SVG path representation of the letter using polylines with visual markers.

        This method converts the letter's path to straight lines and adds markers to visualize
        the path structure, making it ideal for debugging complex letter shapes and understanding
        control point relationships in curves.

        Command conversion:
            M, L, Z: Preserved as-is (move-to, line-to, close-path)
            Q: Converted to lines connecting all three points (start, control, end)
            C: Converted to lines connecting all four points (start, control1, control2, end)

        Visual markers (size based on stroke_width):
            - Squares: Path points (L commands and curve endpoints)
            - Circles: Control points (intermediate points in Q and C commands)
            - Right triangles: Move-to points (M commands - segment starts)
            - Left triangles: Points before close-path (Z commands - segment ends)

        Args:
            stroke_width: Determines marker sizes (default: 1.0)

        Returns:
            str: Complete SVG path string with polylines and markers.
                    Uses the letter's scale and position for positioning.
        """
        return self._glyph.path.svg_path_string_debug_polyline(self.scale, self.xpos, self.ypos, stroke_width)


###############################################################################
# MultiWeightLetter
###############################################################################


@dataclass
class AvMultiWeightLetter(AvLetter):
    """
    MultiWeightLetter: Container for managing collections of AvSingleGlyphLetter objects with weight support.
    Can handle multiple letters of the same character with different weights, all at the same position.

    Weight Convention:
    - Factories/letters are ordered from HEAVIEST to LIGHTEST (index 0 = heavy/black)
    - Normalized weights: 0.0 = lightest (white), 1.0 = heaviest (black)
    - Index 0 corresponds to normalized weight 1.0 (heaviest)
    - Index N-1 corresponds to normalized weight 0.0 (lightest)
    - This matches image convention: black (0) = heavy, white (255) = light
    """

    _letters: List[AvSingleGlyphLetter] = field(default_factory=list)

    def __init__(
        self,
        letters: List[AvSingleGlyphLetter],
    ) -> None:
        if letters:
            # Get scale, xpos, ypos from the first letter
            first_letter = letters[0]
            super().__init__(
                scale=first_letter.scale,
                xpos=first_letter.xpos,
                ypos=first_letter.ypos,
            )
        else:
            # Initialize with default values for empty list
            super().__init__(scale=1.0, xpos=0.0, ypos=0.0)

        self._letters = letters

        # Validate all letters are AvSingleGlyphLetter objects
        for i, letter in enumerate(self._letters):
            if not isinstance(letter, AvSingleGlyphLetter):
                raise TypeError(f"Letter at index {i} is not an AvSingleGlyphLetter object")

        # Validate all letters have the same character (for multi-weight use case)
        if len(self._letters) > 1:
            first_char = self._letters[0]._glyph.character
            for i, letter in enumerate(self._letters[1:], 1):
                glyph = letter._glyph
                if glyph.character != first_char:
                    print(
                        f"Warning: Letter at index {i} has character '{glyph.character}' "
                        f"but expected '{first_char}' for multi-weight letter"
                    )

    @classmethod
    def from_factories(
        cls,
        character: str,
        factories: List[AvGlyphFactory],
        scale: float = 1.0,
        xpos: float = 0.0,
        ypos: float = 0.0,
    ) -> AvMultiWeightLetter:
        """
        Create AvMultiWeightLetter from multiple factories.

        Weight Convention:
        - Factories must be ordered from HEAVIEST to LIGHTEST
        - Index 0 = heaviest weight (black, normalized weight 1.0)
        - Index N-1 = lightest weight (white, normalized weight 0.0)
        - Normalized weights are evenly spaced: [1.0, ..., 0.0]

        Args:
            character: The character to create glyphs for
            factories: List of factories (ordered from HEAVIEST to LIGHTEST weight)
            scale: Scale factor
            xpos: X position
            ypos: Y position
        """
        letters = []

        for factory in factories:
            glyph = factory.get_glyph(character)
            letter = AvSingleGlyphLetter(glyph=glyph, scale=scale, xpos=xpos, ypos=ypos)
            letters.append(letter)

        return cls(letters=letters)

    ####################################################################################

    @property
    def xpos(self) -> float:
        """The x position of the letter in real dimensions."""
        return super().xpos

    @xpos.setter
    def xpos(self, xpos: float) -> None:
        """Sets the x position of the letter in real dimensions."""
        delta_xpos = xpos - self.xpos
        for letter in self._letters:
            letter.xpos = letter.xpos + delta_xpos
        self._xpos = xpos

    @property
    def ypos(self) -> float:
        """The y position of the letter in real dimensions."""
        return super().ypos

    @ypos.setter
    def ypos(self, ypos: float) -> None:
        """Sets the y position of the letter in real dimensions."""
        delta_ypos = ypos - self.ypos
        for letter in self._letters:
            letter.ypos = letter.ypos + delta_ypos
        self._ypos = ypos

    ####################################################################################

    @property
    def advance_width(self) -> float:
        """Returns the advance width of the combined letter box."""
        if not self._letters:
            return 0.0
        return self.letter_box.width

    @property
    def bounding_box(self) -> AvBox:
        """Returns the combined bounding box around all letters' outlines.

        Combines transformed bounding_boxes from all letters using AvBox.combine.
        Returns empty box if no letters.

        Returns:
            AvBox: Combined bounding box of all letters.
        """
        if not self._letters:
            return AvBox(0.0, 0.0, 0.0, 0.0)

        # Get bounding boxes from all letters
        bounding_boxes = [letter.bounding_box for letter in self._letters]

        # Use AvBox.combine to get the overall bounding box
        return AvBox.combine(*bounding_boxes)

    @property
    def letter_box(self) -> AvBox:
        """Returns the combined letter box around all letters' advance boxes.

        Combines transformed glyph_boxes (AvBox(0, descender, advance_width, ascender))
        from all letters using AvBox.combine. Includes full advance widths of all letters.
        Returns empty box if no letters.

        Returns:
            AvBox: Combined letter box of all letters.
        """
        if not self._letters:
            return AvBox(0.0, 0.0, 0.0, 0.0)

        # Get letter boxes from all letters
        letter_boxes = [letter.letter_box for letter in self._letters]

        # Use AvBox.combine to get the overall letter box
        return AvBox.combine(*letter_boxes)

    def polygonize_path(self, steps: int = 50) -> Optional[AvPath]:
        """Return the polygonized outline of the heaviest letter.

        Args:
            steps: Number of segments for curve approximation (default: 50).

        Returns:
            Polygonized path in world coordinates from the heaviest
            (index 0) letter, or ``None`` if there are no letters.
        """
        if not self._letters:
            return None
        return self._letters[0].polygonize_path(steps)

    def svg_path_string(self) -> str:
        """SVG path string of the letter in real dimensions."""
        if not self._letters:
            return "M 0 0"

        # Merge all letter path strings
        path_strings = []
        for letter in self._letters:
            path_strings.append(letter.svg_path_string())

        return " ".join(path_strings)

    @property
    def centroid(self) -> Tuple[float, float]:
        """Returns the centroid of the heaviest letter in real dimensions.

        Returns the centroid from the first (heaviest) letter.

        Returns:
            Tuple[float, float]: The (x, y) coordinates of the centroid.
        """
        if not self._letters:
            return (0.0, 0.0)
        return self._letters[0].centroid

    def exterior(self, steps: int = 20) -> List[AvSinglePolygonPath]:
        """Get the exterior paths of the heaviest letter without holes.

        Returns the exterior paths from the first (heaviest) letter.

        Args:
            steps: Number of segments to use for curve approximation during polygonization

        Returns:
            List of AvSinglePolygonPath objects representing only positive polygons
            (exterior rings without any holes) in the letter's coordinate system
        """
        if not self._letters:
            return []
        return self._letters[0].exterior(steps)

    @property
    def exterior_path(self) -> List[AvSinglePolygonPath]:
        """Convert glyph outline to one or more polygons without holes using fixed internal steps.

        Returns:
            List of transformed AvSinglePolygonPath objects representing only positive polygons
            (exterior rings without any holes) in the letter's coordinate system
        """
        if not self._letters:
            return []
        return self._letters[0].exterior_path

    @property
    def exterior_path_left_silhouette(self) -> List[AvSinglePolygonPath]:
        """Get left orthographic silhouette of the letter's exterior polygons.

        Returns:
            List of AvSinglePolygonPath objects representing the left silhouette
            with right blocking edge at x = max_x in the letter's coordinate system
        """
        if not self._letters:
            return []
        return self._letters[0].exterior_path_left_silhouette

    @property
    def exterior_path_right_silhouette(self) -> List[AvSinglePolygonPath]:
        """Get right orthographic silhouette of the letter's exterior polygons.

        Returns:
            List of AvSinglePolygonPath objects representing the right silhouette
            with left blocking edge at x = min_x in the letter's coordinate system
        """
        if not self._letters:
            return []
        return self._letters[0].exterior_path_right_silhouette

    ####################################################################################

    @property
    def letters(self) -> List[AvSingleGlyphLetter]:
        """Get the letters list."""
        return self._letters

    @property
    def character(self) -> str:
        """Get the character (from first letter)."""
        if not self._letters:
            return ""
        return self._letters[0].glyph.character

    @property
    def num_weights(self) -> int:
        """Get the number of weight variants.

        Returns:
            Number of weight variants (letters) in this multi-weight letter.
        """
        return len(self._letters)

    def get_normalized_weight(self, index: int) -> float:
        """Get the normalized weight value for a given index.

        Weight Convention:
        - Index 0 = heaviest (normalized weight 1.0)
        - Index N-1 = lightest (normalized weight 0.0)
        - Weights are evenly spaced between 1.0 and 0.0

        Args:
            index: Letter index (0 to N-1)

        Returns:
            Normalized weight in range [0.0, 1.0] where 1.0 = heaviest, 0.0 = lightest

        Raises:
            IndexError: If index is out of range
        """
        if not 0 <= index < len(self._letters):
            raise IndexError(f"Index {index} out of range [0, {len(self._letters) - 1}]")

        if len(self._letters) == 1:
            return 1.0  # Single weight is considered heaviest

        # Index 0 = weight 1.0 (heaviest), Index N-1 = weight 0.0 (lightest)
        return 1.0 - (index / (len(self._letters) - 1))

    def get_letter_by_weight(self, normalized_weight: float) -> AvSingleGlyphLetter:
        """Get the letter closest to the specified normalized weight.

        Weight Convention:
        - normalized_weight 1.0 = heaviest (black) → returns letter at index 0
        - normalized_weight 0.0 = lightest (white) → returns letter at index N-1
        - Intermediate values interpolate and round to nearest available weight

        Args:
            normalized_weight: Weight value in range [0.0, 1.0]

        Returns:
            The letter with the closest matching weight

        Raises:
            ValueError: If normalized_weight is out of range or no letters available
        """
        if not self._letters:
            raise ValueError("No letters available")

        if not 0.0 <= normalized_weight <= 1.0:
            raise ValueError(f"normalized_weight must be in [0.0, 1.0], got {normalized_weight}")

        if len(self._letters) == 1:
            return self._letters[0]

        # Convert normalized weight to index
        # weight 1.0 → index 0, weight 0.0 → index N-1
        index = int((1.0 - normalized_weight) * (len(self._letters) - 1) + 0.5)
        index = max(0, min(index, len(self._letters) - 1))  # Clamp to valid range

        return self._letters[index]

    @staticmethod
    def gray_to_weight_index(gray_value: float, num_weights: int) -> int:
        """Convert image gray value (0-255) to weight index.

        Maps image gray values to glyph weight indices consistently:
        - Black (0) → index 0 (heaviest weight)
        - White (255) → index N-1 (lightest weight)
        - Gray values interpolate linearly

        Args:
            gray_value: Gray value from image (0=black, 255=white)
            num_weights: Number of available weight variants

        Returns:
            Weight index (0=heaviest, N-1=lightest)

        Raises:
            ValueError: If parameters are out of valid range

        Example:
            >>> AvMultiWeightLetter.gray_to_weight_index(0, 9)    # black
            0  # heaviest
            >>> AvMultiWeightLetter.gray_to_weight_index(255, 9)  # white
            8  # lightest
            >>> AvMultiWeightLetter.gray_to_weight_index(127.5, 9)  # mid-gray
            4  # medium weight
        """
        if not 0 <= gray_value <= 255:
            raise ValueError(f"gray_value must be in [0, 255], got {gray_value}")
        if num_weights < 1:
            raise ValueError(f"num_weights must be >= 1, got {num_weights}")

        if num_weights == 1:
            return 0

        # Direct mapping: darker (lower) values → lower indices (heavier)
        normalized = gray_value / 255.0  # 0.0 to 1.0 (0=black, 1=white)
        index = int(normalized * (num_weights - 1) + 0.5)  # Round to nearest
        return min(index, num_weights - 1)  # Clamp to valid range

    @staticmethod
    def gray_normalized_to_weight_index(gray_normalized: float, num_weights: int) -> int:
        """Convert normalized gray value (0.0-1.0) to weight index.

        Maps normalized gray values to glyph weight indices:
        - Black (0.0) → index 0 (heaviest weight)
        - White (1.0) → index N-1 (lightest weight)
        - Gray values interpolate linearly

        Args:
            gray_normalized: Normalized gray value (0.0=black, 1.0=white)
            num_weights: Number of available weight variants

        Returns:
            Weight index (0=heaviest, N-1=lightest)

        Raises:
            ValueError: If parameters are out of valid range

        Example:
            >>> AvMultiWeightLetter.gray_normalized_to_weight_index(0.0, 9)  # black
            0  # heaviest
            >>> AvMultiWeightLetter.gray_normalized_to_weight_index(1.0, 9)  # white
            8  # lightest
            >>> AvMultiWeightLetter.gray_normalized_to_weight_index(0.5, 9)  # mid-gray
            4  # medium weight
        """
        if not 0.0 <= gray_normalized <= 1.0:
            raise ValueError(f"gray_normalized must be in [0.0, 1.0], got {gray_normalized}")
        if num_weights < 1:
            raise ValueError(f"num_weights must be >= 1, got {num_weights}")

        if num_weights == 1:
            return 0

        # Direct mapping: darker (lower) values → lower indices (heavier)
        index = int(gray_normalized * (num_weights - 1) + 0.5)  # Round to nearest
        return min(index, num_weights - 1)  # Clamp to valid range


###############################################################################
# Letter Factories
###############################################################################

###############################################################################
# AvLetterFactory
###############################################################################


class AvLetterFactory(ABC):
    """Abstract base class for letter factories.

    Letter factories create AvLetter instances from stream items (characters or syllables).
    This class-based approach allows for stateful factories and better extensibility
    compared to simple callables.
    """

    @abstractmethod
    def create_letter(
        self,
        item: str,
        scale: float,
        xpos: float,
        ypos: float,
    ) -> AvLetter:
        """Create a letter from the given parameters.

        Args:
            item: The item from the stream (character or syllable)
            scale: Scale factor for the letter
            xpos: X position of the letter
            ypos: Y position of the letter

        Returns:
            An AvLetter instance
        """
        raise NotImplementedError


###############################################################################
# AvSingleGlyphLetterFactory
###############################################################################


class AvSingleGlyphLetterFactory(AvLetterFactory):
    """Factory for creating AvSingleGlyphLetter objects.

    This factory uses a glyph factory to create single glyph letters.
    It's the most common factory type for simple text layout.
    """

    def __init__(self, glyph_factory: AvGlyphFactory):
        """Initialize the factory with a glyph factory.

        Args:
            glyph_factory: Factory to create glyphs for characters
        """
        self._glyph_factory = glyph_factory

    def create_letter(
        self,
        item: str,
        scale: float,
        xpos: float,
        ypos: float,
    ) -> AvLetter:
        """Create an AvSingleGlyphLetter from the given parameters.

        Args:
            item: The character to create a letter for
            scale: Scale factor for the letter
            xpos: X position of the letter
            ypos: Y position of the letter

        Returns:
            An AvSingleGlyphLetter instance
        """
        glyph = self._glyph_factory.get_glyph(item)
        return AvSingleGlyphLetter(
            glyph=glyph,
            scale=scale,
            xpos=xpos,
            ypos=ypos,
        )


###############################################################################
# AvMultiWeightLetterFactory
###############################################################################


class AvMultiWeightLetterFactory(AvLetterFactory):
    """Factory for creating AvMultiWeightLetter objects.

    This factory uses multiple glyph factories to create letters with different
    weights (e.g., light, regular, bold). Each glyph factory represents a
    different weight, and the resulting letter contains all weight variants.
    """

    def __init__(self, glyph_factories: List[AvGlyphFactory]):
        """Initialize the factory with multiple glyph factories.

        Args:
            glyph_factories: List of glyph factories for different weights
        """
        self._glyph_factories = glyph_factories

    def create_letter(
        self,
        item: str,
        scale: float,
        xpos: float,
        ypos: float,
    ) -> AvLetter:
        """Create an AvMultiWeightLetter from the given parameters.

        Args:
            item: The character to create letters for
            scale: Scale factor for the letters
            xpos: X position of the letters
            ypos: Y position of the letters

        Returns:
            An AvMultiWeightLetter instance containing all weight variants
        """
        letters = []
        for glyph_factory in self._glyph_factories:
            glyph = glyph_factory.get_glyph(item)
            letter = AvSingleGlyphLetter(
                glyph=glyph,
                scale=scale,
                xpos=xpos,
                ypos=ypos,
            )
            letters.append(letter)

        return AvMultiWeightLetter(letters=letters)
