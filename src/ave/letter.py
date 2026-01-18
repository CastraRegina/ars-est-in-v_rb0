"""Multi-glyph container for managing collections of AvGlyph objects."""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ave.common import Align
from ave.geom import AvBox
from ave.glyph import (
    AvGlyph,
    AvGlyphCachedFactory,
    AvGlyphFactory,
    AvGlyphPersistentFactory,
)

###############################################################################
# AvLetter
###############################################################################


@dataclass
class AvLetter:
    """
    A Letter is a Glyph which is scaled to real dimensions with a position, alignment and horizontal offset.
    """

    _glyph: AvGlyph
    _scale: float  # scale from glyph-coordinates to real dimensions (font_size / units_per_em)
    _xpos: float  # left-to-right, value in real dimensions
    _ypos: float  # bottom-to-top, value in real dimensions
    _align: Optional[Align] = None  # LEFT, RIGHT, BOTH. Defaults to None.
    _x_offset: float = 0.0  # horizontal offset in real dimensions

    def __init__(
        self,
        glyph: AvGlyph,
        scale: float = 1.0,  # scale from glyph-coordinates to real dimensions (font_size / units_per_em)
        xpos: float = 0.0,  # left-to-right, value in real dimensions
        ypos: float = 0.0,  # bottom-to-top, value in real dimensions
        align: Optional[Align] = None,  # LEFT, RIGHT, BOTH. Defaults to None.
        x_offset: float = 0.0,  # horizontal offset in real dimensions
    ) -> None:
        self._glyph = glyph
        self._scale = scale
        self._xpos = xpos
        self._ypos = ypos
        self._align = align
        self._x_offset = x_offset

    @property
    def glyph(self) -> AvGlyph:
        """The glyph of the letter."""
        return self._glyph

    @property
    def scale(self) -> float:
        """Returns the scale factor for the letter which is used to transform the glyph to real dimensions."""
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        """Sets the scale factor for the letter which is used to transform the glyph to real dimensions."""
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
    def align(self) -> Optional[Align]:
        """Returns the alignment of the letter; None, LEFT, RIGHT, BOTH."""
        return self._align

    @align.setter
    def align(self, align: Optional[Align]) -> None:
        """Sets the alignment of the letter; None, LEFT, RIGHT, BOTH."""
        self._align = align

    @property
    def x_offset(self) -> float:
        """Horizontal offset in real dimensions."""
        return self._x_offset

    @x_offset.setter
    def x_offset(self, x_offset: float) -> None:
        """Sets the horizontal offset in real dimensions."""
        self._x_offset = x_offset

    @property
    def trafo(self) -> List[float]:
        """
        Returns the affine transformation matrix for the letter to transform the glyph to real dimensions.
        Returns:    [scale, 0, 0, scale, xpos + x_offset, ypos] or
                    [scale, 0, 0, scale, xpos - lsb + x_offset, ypos] if alignment is LEFT or BOTH.
        """
        if self.align in (Align.LEFT, Align.BOTH):
            lsb_scaled = self.scale * self._glyph.left_side_bearing
            return [self.scale, 0, 0, self.scale, self.xpos - lsb_scaled + self._x_offset, self.ypos]
        return [self.scale, 0, 0, self.scale, self.xpos + self._x_offset, self.ypos]

    def width(self, consider_x_offset: bool = False) -> float:
        """
        Returns the width calculated considering the alignment.

        Args:
            consider_x_offset: If True, includes the horizontal offset in the calculation
        """
        if consider_x_offset:
            return self.scale * self._glyph.width(self.align) + self._x_offset
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
        LSB: The horizontal space on the left side of the Letter taking alignment into account (sign varies +/-).
        """
        if self.align in (Align.LEFT, Align.BOTH):
            return 0.0
        return self.scale * self._glyph.left_side_bearing

    @property
    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of the Letter taking alignment into account (sign varies +/-).
        """
        if self.align in (Align.RIGHT, Align.BOTH):
            return 0.0
        return self.scale * self._glyph.right_side_bearing

    def bounding_box(self) -> AvBox:
        """
        Returns bounding box (tightest box around Letter) in real dimensions.
        Coordinates are relative to baseline-origin (0,0) with orientation left-to-right, bottom-to-top
        Returns:
            AvBox: The bounding box of the letter.
        """
        return self._glyph.bounding_box().transform_affine(self.trafo)

    def letter_box(self) -> AvBox:
        """
        Returns the box of the letter in real dimensions.

        This box is the glyph box, which is the glyph’s advance box, not the outline bounding box.
        It is defined by x = 0 to advanceWidth and y = descender to ascender.
        The glyph box is transformed using the trafo of the letter.

        Returns:
            AvBox: The letter box, i.e. the glyph box after transformation.
        """
        return self._glyph.glyph_box().transform_affine(self.trafo)

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
        glyph_centroid = self._glyph.centroid()
        scale, _, _, _, translate_x, translate_y = self.trafo

        # Transform the centroid coordinates
        centroid_x = glyph_centroid[0] * scale + translate_x
        centroid_y = glyph_centroid[1] * scale + translate_y

        return (centroid_x, centroid_y)

    def svg_path_string(self) -> str:
        """
        Returns the SVG path representation of the letter in real dimensions.
        The SVG path is a string that defines the outline of the letter using
        SVG path commands. This path can be used to render the letter as a
        vector graphic.
        Returns:
            str: The SVG path string representing the letter.
        """
        scale, _, _, _, translate_x, translate_y = self.trafo
        return self._glyph.path.svg_path_string(scale, translate_x, translate_y)

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
            stroke_width (float): The stroke width used to determine marker sizes.

        Returns:
            str: The debug SVG path string using only polylines with markers.
                Markers include:
                - Squares: Regular points (L commands)
                - Circles: Control points (intermediate points in Q and C commands)
                - Triangles (pointing right): M command points (segment starts)
                - Triangles (pointing left): Points before Z commands (segment ends)
                Note: Triangles are drawn in addition to the base markers (squares/circles).
        """
        scale, _, _, _, translate_x, translate_y = self.trafo
        return self._glyph.path.svg_path_string_debug_polyline(scale, translate_x, translate_y, stroke_width)


###############################################################################
# MultiWeightLetter
###############################################################################


@dataclass
class AvMultiWeightLetter:
    """
    MultiWeightLetter: Container for managing collections of AvLetter objects with weight support.
    Can handle multiple letters of the same character with different weights, all at the same position.
    Internal weights are normalized from 0 to 1 with equal spacing.
    """

    _letters: List[AvLetter] = field(default_factory=list)

    def __init__(
        self,
        letters: List[AvLetter],
    ) -> None:
        self._letters = letters

        # Validate all letters are AvLetter objects
        for i, letter in enumerate(self._letters):
            if not isinstance(letter, AvLetter):
                raise TypeError(f"Letter at index {i} is not an AvLetter object")

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
        align: Optional[Align] = None,
        x_offsets: Optional[List[float]] = None,
    ) -> "AvMultiWeightLetter":
        """
        Create AvMultiWeightLetter from multiple factories.
        Internal weights will be automatically assigned from 0 to 1 with equal spacing.

        Args:
            character: The character to create glyphs for
            factories: List of factories (ordered from lightest to heaviest weight)
            scale: Scale factor
            xpos: X position
            ypos: Y position
            align: Alignment
            x_offsets: Optional X offsets for each glyph (defaults to all 0.0 for stacked)
        """
        letters = []

        if x_offsets is None:
            x_offsets = [0.0] * len(factories)  # Stack all at same position
        elif len(x_offsets) != len(factories):
            raise ValueError("x_offsets must have the same length as factories")

        for factory, x_offset in zip(factories, x_offsets):
            glyph = factory.get_glyph(character)
            letter = AvLetter(glyph=glyph, scale=scale, xpos=xpos, ypos=ypos, align=align, x_offset=x_offset)
            letters.append(letter)

        return cls(letters=letters)

    @property
    def character(self) -> str:
        """Get the character (from first letter)."""
        if not self._letters:
            return ""
        return self._letters[0].glyph.character

    @property
    def xpos(self) -> float:
        """The x position of the letter in real dimensions."""
        if not self._letters:
            return 0.0
        return self._letters[0].xpos

    @xpos.setter
    def xpos(self, xpos: float) -> None:
        """Sets the x position of the letter in real dimensions."""
        for letter in self._letters:
            letter.xpos = xpos

    @property
    def ypos(self) -> float:
        """The y position of the letter in real dimensions."""
        if not self._letters:
            return 0.0
        return self._letters[0].ypos

    @ypos.setter
    def ypos(self, ypos: float) -> None:
        """Sets the y position of the letter in real dimensions."""
        for letter in self._letters:
            letter.ypos = ypos

    @property
    def scale(self) -> float:
        """Returns the scale factor for the letter which is used to transform the glyph to real dimensions."""
        if not self._letters:
            return 1.0
        return self._letters[0].scale

    @scale.setter
    def scale(self, scale: float) -> None:
        """Sets the scale factor for the letter."""
        for letter in self._letters:
            letter.scale = scale

    @property
    def align(self) -> Optional[Align]:
        """The alignment of the letter; None, LEFT, RIGHT, BOTH."""
        if not self._letters:
            return None
        return self._letters[0].align

    @align.setter
    def align(self, align: Optional[Align]) -> None:
        """Sets the alignment of the letter."""
        for letter in self._letters:
            letter.align = align

    @property
    def letters(self) -> List[AvLetter]:
        """Get the letters list."""
        return self._letters

    @property
    def x_offsets(self) -> List[float]:
        """Get the x offsets for each letter."""
        return [letter.x_offset for letter in self._letters]

    @property
    def glyphs(self) -> List[AvGlyph]:
        """Get the glyphs list."""
        return [letter.glyph for letter in self._letters]

    def width(self) -> float:
        """
        Returns the width calculated considering the alignment.
        """
        if len(self._letters) == 0:
            return 0.0
        return self._letters[0].width(consider_x_offset=True)

    # @property
    # def height(self) -> float:
    # --- not yet implemented ---

    @property
    def ascender(self) -> float:
        """
        The maximum distance above the baseline, i.e. the highest y-coordinate of a Letter (positive value).
        """
        if not self._letters:
            return 0.0
        return max(letter.ascender for letter in self._letters)

    @property
    def descender(self) -> float:
        """
        The maximum distance below the baseline, i.e. the lowest y-coordinate of a Letter (negative value).
        """
        if not self._letters:
            return 0.0
        return min(letter.descender for letter in self._letters)

    # TODO: check implementation
    @property
    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of the Letter taking alignment into account.
        """
        if len(self._letters) == 0:
            return 0.0
        return self._letters[0].left_side_bearing

    # TODO: check implementation
    @property
    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of the Letter taking alignment into account.
        """
        if len(self._letters) == 0:
            return 0.0
        return self._letters[0].right_side_bearing

    def bounding_box(self):
        """Get bounding box that encompasses all letters."""
        if not self._letters:
            return AvBox(0.0, 0.0, 0.0, 0.0)

        # Get bounding boxes from all letters
        letter_bounding_boxes = [letter.bounding_box() for letter in self._letters]

        # Use AvBox.combine to get the overall bounding box
        return AvBox.combine(*letter_bounding_boxes)

    def letter_box(self) -> AvBox:
        """Get letter box that encompasses all letters."""
        if not self._letters:
            return AvBox(0.0, 0.0, 0.0, 0.0)

        # Get letter boxes from all letters
        letter_letter_boxes = [letter.letter_box() for letter in self._letters]

        # Use AvBox.combine to get the overall letter box
        return AvBox.combine(*letter_letter_boxes)

    def svg_path_string(self) -> str:
        """SVG path string of the letter in real dimensions."""
        if not self._letters:
            return "M 0 0"

        # Merge all letter path strings
        path_strings = []
        for letter in self._letters:
            path_strings.append(letter.svg_path_string())

        return " ".join(path_strings)

    @staticmethod
    def adapt_x_offset_by_centroid(multi_weight_letter: AvMultiWeightLetter) -> None:
        """
        Adapt x_offset of all letters so their bounding box centers align.

        Uses the bounding box center (visual bounds) rather than centroid or
        pole of inaccessibility, as it provides the most robust and consistent
        alignment for all character types.

        Args:
            multi_weight_letter: The AvMultiWeightLetter to modify
        """
        if not multi_weight_letter.letters or len(multi_weight_letter.letters) < 2:
            return

        # Get the heaviest letter (last in the list) as reference
        reference_letter = multi_weight_letter.letters[-1]
        reference_bbox = reference_letter.bounding_box()
        reference_center_x: float = (reference_bbox.xmin + reference_bbox.xmax) / 2

        # Adjust x_offset for all letters except the reference
        for letter in multi_weight_letter.letters[:-1]:
            bbox = letter.bounding_box()
            center_x: float = (bbox.xmin + bbox.xmax) / 2

            # Calculate the offset needed to align centers
            offset_adjustment: float = reference_center_x - center_x

            # Update the letter's x_offset
            letter.x_offset += offset_adjustment

    @staticmethod
    def adapt_x_offset_by_overlap(
        multi_weight_letter: AvMultiWeightLetter,
        step_factor: float = 0.01,
        max_iter: int = 10,
        num_samples: int = 10,
    ) -> None:
        """
        Align the horizontal alignment of multi-weight letters by minimizing overlap.

        algorithm iteratively adjusts the horizontal offset of lighter-weight letters
        to minimize the horizontal distance between their outlines and the heaviest letter.
        Starting from centroid alignment, it tries small horizontal shifts to minimise a cost
        function based on the horizontal distance between the outlines. It works robustly
        even when glyphs have different numbers of points or slightly different shapes.

        The algorithm:
        1. Starts with centroid alignment as initial guess
        2. Samples y-positions across the glyph height
        3. For each sample, finds horizontal intersections with the outline
        4 Compares left/right edges between reference and other letters
        4. Iteratively adjusts offsets to minimise the total edge distance

        Args:
            multi_weight_letter: The AvMultiWeightLetter to modify.
            step_factor: Step size as fraction of glyph width (default 1% of width).
            max_iter: Upper bound on the number of optimisation passes.
            num_samples: Number of y-positions to sample for cost evaluation.
        """
        if not multi_weight_letter.letters or len(multi_weight_letter.letters) < 2:
            return

        # Start with centroid alignment as initial guess
        AvMultiWeightLetter.adapt_x_offset_by_centroid(multi_weight_letter)

        # The heaviest glyph stays fixed as reference
        reference = multi_weight_letter.letters[-1]

        # Calculate step size based on reference glyph width
        ref_bbox = reference.bounding_box()
        step = step_factor * (ref_bbox.xmax - ref_bbox.xmin)

        # Pre-compute sample y-positions within the reference's vertical bounds
        y_min = ref_bbox.ymin
        y_max = ref_bbox.ymax
        y_samples = np.linspace(y_min, y_max, num_samples)

        # Helper: get horizontal intersections for a letter at given y
        def _horizontal_intersections(letter: AvLetter, y: float) -> Tuple[float, float]:
            """
            Get the leftmost and rightmost x-coordinates of the letter outline at a given y.
            Returns a tuple (x_left, x_right).
            If the letter has no outline at this y, returns (inf, -inf).
            """
            # Transform y to glyph space
            scale, _, _, _, translate_x, translate_y = letter.trafo
            y_glyph = (y - translate_y) / scale

            # Get the glyph path
            path = letter.glyph.path

            # Check if path is polygon-like, otherwise polygonize
            if path.is_polygon_like:
                points = path.points
            else:
                points = path.polygonized_path().points

            # Find all segments that cross the y-coordinate
            x_coords = []
            n = len(points)
            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]

                # Check if segment crosses y (in glyph space)
                if min(p1[1], p2[1]) <= y_glyph <= max(p1[1], p2[1]):
                    # Linear interpolation to find x at this y
                    if abs(p2[1] - p1[1]) > 1e-10:
                        t = (y_glyph - p1[1]) / (p2[1] - p1[1])
                        x = p1[0] + t * (p2[0] - p1[0])
                        x_coords.append(x)

            if x_coords:
                # Transform back to real space
                x_coords_real = [x * scale + translate_x for x in x_coords]
                return min(x_coords_real), max(x_coords_real)
            else:
                return float("inf"), float("-inf")

        # Helper: compute overlap cost between reference and another letter
        def _compute_cost(letter: AvLetter) -> float:
            """Compute horizontal overlap cost as sum of edge differences."""
            cost = 0.0
            for y in y_samples:
                ref_left, ref_right = _horizontal_intersections(reference, y)
                other_left, other_right = _horizontal_intersections(letter, y)

                # Handle cases where one glyph is inside the other
                if ref_left != float("inf") and other_left != float("inf"):
                    # If one glyph is completely inside the other, compare centers
                    if ref_left <= other_left and other_right <= ref_right:
                        # Other glyph is inside reference - penalize center misalignment
                        ref_center = (ref_left + ref_right) / 2
                        other_center = (other_left + other_right) / 2
                        cost += abs(ref_center - other_center) * 2  # Double penalty for center misalignment
                    elif other_left <= ref_left and ref_right <= other_right:
                        # Reference is inside other - penalize center misalignment
                        ref_center = (ref_left + ref_right) / 2
                        other_center = (other_left + other_right) / 2
                        cost += abs(ref_center - other_center) * 2
                    else:
                        # Partial overlap - compare edges
                        cost += abs(ref_left - other_left) + abs(ref_right - other_right)
                elif ref_left == float("inf") and other_left != float("inf"):
                    # Only reference has outline at this y - penalize heavily
                    cost += 100.0
                elif ref_left != float("inf") and other_left == float("inf"):
                    # Only other has outline at this y - penalize heavily
                    cost += 100.0
            return cost

        # Iterative optimisation
        for _ in range(max_iter):
            improved = False

            for letter in multi_weight_letter.letters[:-1]:
                current_cost = _compute_cost(letter)

                # Try moving left and right by step
                for delta in (-step, step):
                    letter.x_offset += delta
                    new_cost = _compute_cost(letter)

                    if new_cost < current_cost:
                        # Keep the improvement
                        improved = True
                        current_cost = new_cost
                    else:
                        # Revert the trial move
                        letter.x_offset -= delta

            if not improved:
                # No further improvement - stop early
                break


def main():
    """Test function for AvMultiWeightLetter."""
    from ave.page import AvSvgPage  # pylint: disable=import-outside-toplevel

    def load_cached_fonts(path_name: str, font_fn_base: str) -> List[AvGlyphCachedFactory]:
        """
        Load cached font files from directory.

        Args:
            path_name: Directory path where cached files are stored (e.g., "fonts/cache")
            font_fn_base: Base filename pattern (e.g., "RobotoFlex-VariableFont_AA_")

        Returns:
            List[AvGlyphCachedFactory]: Ordered list of font factories (lightest to heaviest)
        """
        cache_dir = Path(path_name)
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {path_name}")

        # Find all matching files
        pattern = f"{font_fn_base}*_wght*.json.zip"
        file_paths = list(cache_dir.glob(pattern))

        if not file_paths:
            raise FileNotFoundError(f"No cached font files found with pattern: {pattern}")

        # Load factories (files are already in light to heavy order)
        factories = []

        for file_path in file_paths:
            try:
                factory = AvGlyphPersistentFactory.load_from_file(str(file_path))
                factories.append(factory)
                print(f"Loaded: {file_path.name}")
            except (FileNotFoundError, ValueError, RuntimeError, OSError, gzip.BadGzipFile) as e:
                print(f"Warning: Failed to load {file_path.name}: {e}")

        if not factories:
            raise ValueError("No valid cached font files could be loaded")

        return factories

    # Example usage
    try:
        factories = load_cached_fonts("fonts/cache", "RobotoFlex-VariableFont_AA_")
        print(f"Loaded {len(factories)} fonts")

        # Print font properties for each factory
        for i, factory in enumerate(factories):
            font_props = factory.get_font_properties()
            print(
                f"Font {i+1}: "
                f"ascender={font_props.ascender}, "
                f"descender={font_props.descender}, "
                f"units_per_em={font_props.units_per_em}"
            )

        # Example: Create AvMultiWeightLetter with different weights stacked
        print("\nCreating AvMultiWeightLetter with stacked weights...")
        multi_letter = AvMultiWeightLetter.from_factories(
            character="I",
            factories=factories,
            scale=50.0 / 2048.0,  # 50pt font
            xpos=100,
            ypos=100,
        )
        print(f"Created multi-weight letter for character '{multi_letter.character}'")
        print(f"Number of weight variants: {len(multi_letter.letters)}")
        print(f"Initial X positions (for fine-tuning): {multi_letter.x_offsets}")

        #######################################################################

        # Characters to display
        characters = ""
        characters += "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        characters += "abcdefghijklmnopqrstuvwxyz "
        characters += "0123456789 "
        # characters += ',.;:+-*#_<> !"§$%&/()=?{}[] '
        # # NON-ASCII EXCEPTION: German characters and special symbols for comprehensive font testing
        # characters += "ÄÖÜ äöü ß€µ@²³~^°\\ '`"

        # Create SVG page for multi-weight letters
        # Setup the page with A4 dimensions
        viewbox_width = 180  # viewbox width in mm
        viewbox_height = 120  # viewbox height in mm
        vb_scale = 1.0 / viewbox_width  # scale viewbox so that x-coordinates are between 0 and 1
        font_size = vb_scale * 2.7  # in mm (already in viewbox units)

        # Get units_per_em from the first factory
        units_per_em = factories[0].get_font_properties().units_per_em if factories else 2048.0
        scale = font_size / units_per_em  # proper scale calculation

        # Create the SVG page
        svg_page = AvSvgPage.create_page_a4(viewbox_width, viewbox_height, vb_scale)

        # Draw viewbox border
        svg_page.add(
            svg_page.drawing.path(
                d=(
                    f"M 0 0 "
                    f"L {vb_scale * viewbox_width} 0 "
                    f"L {vb_scale * viewbox_width} {vb_scale * viewbox_height} "
                    f"L 0 {vb_scale * viewbox_height} "
                    f"Z"
                ),
                stroke="blue",
                stroke_width=0.1 * vb_scale,
                fill="none",
            ),
            False,
        )

        # Render characters with multi-weight
        current_xpos = 0.05
        current_ypos = 0.02
        max_width = 0.95

        for char in characters:
            print(f"Rendering '{char}' with multi-weight...", end="", flush=True)

            # Create multi-weight letter for this character
            multi_letter = AvMultiWeightLetter.from_factories(
                character=char,
                factories=factories,
                scale=scale,  # Use proper scale, not font_size
                xpos=current_xpos,
                ypos=current_ypos,
            )

            AvMultiWeightLetter.adapt_x_offset_by_overlap(multi_letter, step_factor=0.005, max_iter=10, num_samples=12)

            # Modify x positions for visual effect (stack with slight offset)
            if len(multi_letter.letters) >= 3:
                # If you want to override with manual offsets, uncomment below:
                # multi_letter.letters[0].x_offset = 0.0  # Lightest weight
                # multi_letter.letters[1].x_offset = 0.0  # Medium weight
                # multi_letter.letters[2].x_offset = 0.0  # Heaviest weight
                pass

            # Render each weight variant with different opacity
            colors = ["#000000", "#808080", "#E0E0E0"]  # Black to light gray (reversed)
            for i, (letter, color) in enumerate(zip(reversed(multi_letter.letters), colors)):
                # Use the letter directly for its path
                svg_path = svg_page.drawing.path(letter.svg_path_string(), fill=color, stroke="none")
                svg_page.add(svg_path)

            # Move to next position
            current_xpos += multi_letter.width() + 0.005

            # Check if we need to move to next line
            if current_xpos > max_width:
                current_xpos = 0.05
                current_ypos += scale * 1.5  # Use scale instead of font_size

            print(" done")

        # Save the SVG
        svg_filename = "data/output/example/svg/multi_weight_letters.svgz"
        print(f"\nSaving to {svg_filename} ...")
        svg_page.save_as(svg_filename, include_debug_layer=False, pretty=True, compressed=True)
        print(f"Saved to {svg_filename}")

    except (FileNotFoundError, ValueError, RuntimeError, OSError, gzip.BadGzipFile) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
