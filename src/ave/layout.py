"""Line layout functionality for characters and syllables."""

from __future__ import annotations

import importlib
from typing import Any, Iterator, List

from ave.common import AlignX
from ave.geom import GeomMath
from ave.glyph_factory import AvGlyphFactory
from ave.letter import AvLetter, AvLetterFactory, AvSingleGlyphLetterFactory
from ave.letter_processing import AvLetterAlignment
from ave.letter_support import LetterSpacing
from ave.text import AvCharacterStream, AvStreamBase

###############################################################################
# AvCharLineLayouter
###############################################################################


class AvCharLineLayouter:
    """
    Layouts items from a stream into a line with left/right alignment.

    The layouter creates letters using a factory, positions them
    along the baseline, and distributes excess space among letters when needed.
    Supports different letter types (AvSingleGlyphLetter, AvMultiWeightLetter)
    and different stream types (AvCharacterStream, AvSyllableStream).

    Algorithm:
        1. Sequential Placement:
            - Start with current_x = x_start
            - For each item from stream:
                * Create letter at current_x position
                * For first letter: left-align bounding box to x_start using
                    AvLetterAlignment.align_to_x_border(), then advance by width(LEFT)
                * For subsequent letters: advance by width()
                * Stop when next letter would exceed x_end (give it back to stream)
            - Store all placed letters in _letters list

        2. Edge Alignment:
            - First letter is already left-aligned from step 1
            - If more than 1 letter: right-align last letter to x_end using
                AvLetterAlignment.align_to_x_border()

        3. Excess-Space Distribution:
            - Calculate excess space between last two letters:
                excess_space = (last_letter.xpos - left_neighbor.xpos) - left_neighbor.width()
            - Compute delta_x = excess_space / (n - 1) where n = total letters
            - For each inner letter i (1 <= i <= n-2):
                * Adjust position: xpos[i] += delta_x * i
            - This creates uniform spacing where each gap increases by delta_x

        4. Special Cases:
            - Empty stream: Returns empty list
            - 1 letter: Only left-aligned in step 1, no right alignment or distribution
            - 2 letters: Edge alignment only, no distribution (no inner letters)

    Implementation Details:
        - Letters maintain left_neighbor references for width calculations
        - Stream supports rewind() to return unfitted letters
        - Position tracking uses current_x accumulator during placement
        - Final adjustment preserves first/last letter positions while
            interpolating inner letters
    """

    def __init__(
        self,
        x_start: float,
        x_end: float,
        y_baseline: float,
        stream: AvStreamBase,
        letter_factory: AvLetterFactory,
        scale: float = 1.0,
    ) -> None:
        """
        Initialize the line layouter.

        Args:
            x_start: Starting x position for left alignment
            x_end: Ending x position for right alignment
            y_baseline: Y position of the baseline (constant for all letters)
            stream: Stream providing items (characters/syllables) to layout
            letter_factory: Factory to create letters from items.
            scale: Scale factor for letter dimensions (default: 1.0)
        """
        self._x_start = x_start
        self._x_end = x_end
        self._y_baseline = y_baseline
        self._stream = stream
        self._scale = scale
        self._letters: List[AvLetter] = []
        self._letter_factory = letter_factory

    def layout_line(self) -> List[AvLetter]:
        """
        Layout a line of items from the stream.

        Returns:
            List of positioned AvLetter objects representing the line

        Algorithm:
            1. Reset _letters list and set current_x = x_start

            2. Sequential Placement Loop:
                - Get next item from stream
                - Create letter at current_x position
                - Set left_neighbor reference for all but first letter
                - For first letter: left-align to x_start and use width(LEFT)
                - For others: use width() for advancement
                - If letter would exceed x_end (and not first): rewind and break
                - Add letter to list, advance current_x by letter_width

            3. Edge Alignment:
                - If no letters: return empty list
                - If >1 letter: right-align last letter to x_end

            4. Space Distribution:
                - Calculate excess space between last two letters
                - Compute delta_x = excess_space / (n - 1)
                - Adjust inner letters: xpos[i] += delta_x * i
                - This uniformly distributes space across all gaps
        """
        # Reset letters list
        self._letters = []

        # First pass: place letters sequentially and check fit.
        # The first letter is placed left-aligned (bounding_box.xmin = x_start)
        # and advances by width(LEFT); all others advance by width().
        current_x = self._x_start

        try:
            while True:
                # Get next item from stream
                item = self._stream.next_item()

                # Create letter using factory
                letter = self._letter_factory.create_letter(
                    item,
                    self._scale,
                    current_x,
                    self._y_baseline,
                )

                # Set left neighbor if not first letter
                if self._letters:
                    letter.left_letter = self._letters[-1]

                if not self._letters:
                    # First letter: left-align its bounding box to x_start
                    AvLetterAlignment.align_to_x_border(letter, self._x_start, AlignX.LEFT)
                    letter_width = letter.width(AlignX.LEFT)
                else:
                    letter_width = letter.width()

                if current_x + letter_width > self._x_end and self._letters:
                    # Letter doesn't fit, give it back to stream
                    self._stream.rewind(1)
                    break

                # Letter fits, add to list
                self._letters.append(letter)
                current_x = current_x + letter_width

        except StopIteration:
            # No more characters in stream
            pass

        # If no letters placed, return empty list
        if not self._letters:
            return self._letters

        # Adjust last letter for right alignment
        if len(self._letters) > 1:
            AvLetterAlignment.align_to_x_border(self._letters[-1], self._x_end, AlignX.RIGHT)

        # Retrieve the space to be distributed among the letters
        delta_x = 0
        if len(self._letters) > 1:
            last_letter = self._letters[-1]
            left_neighbor = self._letters[-2]
            excess_space = (last_letter.xpos - left_neighbor.xpos) - left_neighbor.width()
            delta_x = excess_space / (len(self._letters) - 1)

        # Apply the calculated delta_x to adjust spacing between letters
        for i in range(1, len(self._letters) - 1):
            self._letters[i].xpos += delta_x * i

        return self._letters


###############################################################################
# AvSyllableLineLayouter
###############################################################################


class AvSyllableLineLayouter(AvCharLineLayouter):
    """
    Convenience layouter for syllable streams.

    This is a specialized version of AvCharLineLayouter that works with syllable
    streams (AvSyllableStream). Each item from the stream is a syllable (which may
    contain multiple characters), and the layouter creates letters for each syllable.

    The layout algorithm is the same as AvCharLineLayouter - it places syllables
    sequentially and distributes excess space among the letters.
    """

    def __init__(
        self,
        x_start: float,
        x_end: float,
        y_baseline: float,
        stream: AvStreamBase,
        letter_factory: AvLetterFactory,
        scale: float = 1.0,
    ) -> None:
        """
        Initialize the syllable line layouter.

        Args:
            x_start: Starting x position for left alignment
            x_end: Ending x position for right alignment
            y_baseline: Y position of the baseline (constant for all letters)
            stream: Stream providing syllables to layout (typically AvSyllableStream)
            letter_factory: Factory to create letters from syllables.
            scale: Scale factor for letter dimensions (default: 1.0)
        """
        super().__init__(
            x_start=x_start,
            x_end=x_end,
            y_baseline=y_baseline,
            stream=stream,
            letter_factory=letter_factory,
            scale=scale,
        )


###############################################################################
# AvTightCharLineLayouter
###############################################################################


class AvTightCharLineLayouter(AvCharLineLayouter):
    """
    Layouts items tightly using geometric collision detection.

    Places each letter (except the first and last of the line) next to the
    left-letter so that they nearly touch.
    """

    def __init__(
        self,
        x_start: float,
        x_end: float,
        y_baseline: float,
        stream: AvStreamBase,
        letter_factory: AvLetterFactory,
        scale: float = 1.0,
        margin: float = 0.0,
    ) -> None:
        """
        Initialize the tight char line layouter.

        Args:
            x_start: Starting x position for left alignment
            x_end: Ending x position for right alignment
            y_baseline: Y position of the baseline (constant for all letters)
            stream: Stream providing characters to layout (typically AvCharacterStream)
            letter_factory: Factory to create letters from characters.
            scale: Scale factor for letter dimensions (default: 1.0)
            margin: Minimum space between letters in world coordinates (default: 0.0).
        """
        super().__init__(
            x_start=x_start,
            x_end=x_end,
            y_baseline=y_baseline,
            stream=stream,
            letter_factory=letter_factory,
            scale=scale,
        )
        self._margin = margin

    @property
    def margin(self) -> float:
        """
        Get the minimum space between letters in world coordinates.

        Returns:
            float: The margin value.
        """
        return self._margin

    @margin.setter
    def margin(self, value: float) -> None:
        """
        Set the minimum space between letters in world coordinates.

        Args:
            value: The margin value. Must be non-negative.
        """
        if value < 0:
            raise ValueError(f"Margin must be non-negative, got {value}")
        self._margin = value

    def layout_line(self) -> List[AvLetter]:
        """
        Layout a line of items tightly.

        Returns:
            List of positioned AvLetter objects representing the line
        """
        # Reset letters list
        self._letters = []

        try:
            while True:
                # Get next item from stream
                item = self._stream.next_item()

                # Position for first letter: left-aligned at x_start
                if not self._letters:
                    current_x = self._x_start
                else:
                    # For subsequent letters, start from current letter's right edge
                    current_x = self._letters[-1].xpos + self._letters[-1].width()

                # Create letter using factory
                letter = self._letter_factory.create_letter(
                    item,
                    self._scale,
                    current_x,
                    self._y_baseline,
                )

                # Ensure first letter uses LEFT alignment immediately so that
                # spacing calculations reference the correct geometry.
                if not self._letters:
                    letter.xpos = self._x_start

                # Apply tight packing if there is a left neighbor
                if self._letters:
                    letter.left_letter = self._letters[-1]

                    # Calculate space to left neighbor and move letter to touch
                    space = LetterSpacing.space_between(letter.left_letter, letter)
                    # Positive space means we can move left, negative means overlap
                    # After touching, add margin to create minimum space
                    letter.xpos -= space - self.margin

                # Check if letter fits within bounds (entire letter must fit)
                if letter.bounding_box.xmax > self._x_end:
                    if not self._letters:
                        # First letter doesn't fit, place it anyway
                        self._letters.append(letter)
                    else:
                        # Letter doesn't fit, give it back to stream
                        self._stream.rewind(1)
                        break

                # Letter fits, add to list
                self._letters.append(letter)

        except StopIteration:
            # No more characters in stream
            pass

        # If no letters placed, return empty list
        if not self._letters:
            return self._letters

        # Save the initial positions before any adjustment
        initial_positions = [letter.xpos for letter in self._letters]

        # Ensure first letter is left-aligned at x_start
        if len(self._letters) > 0:
            AvLetterAlignment.align_to_x_border(self._letters[0], self._x_start, AlignX.LEFT)

        # Save last letter position before adjusting for right alignment
        tight_last_xpos = self._letters[-1].xpos if len(self._letters) > 1 else None

        # Adjust last letter for right alignment
        if len(self._letters) > 1:
            AvLetterAlignment.align_to_x_border(self._letters[-1], self._x_end, AlignX.RIGHT)

        # Distribute added spacing uniformly across all gaps.
        # Interpolate each letter shift from first-letter shift to last-letter shift.
        if len(self._letters) > 2 and tight_last_xpos is not None:
            first_shift = self._letters[0].xpos - initial_positions[0]
            last_shift = self._letters[-1].xpos - initial_positions[-1]
            shift_step = (last_shift - first_shift) / (len(self._letters) - 1)
            for idx in range(1, len(self._letters) - 1):
                self._letters[idx].xpos = initial_positions[idx] + first_shift + idx * shift_step

        elif len(self._letters) == 1:
            # Single letter, center it
            single_letter = self._letters[0]
            center_offset = (self._x_end - self._x_start - single_letter.width()) / 2
            single_letter.xpos = self._x_start + center_offset

        return self._letters


###############################################################################
# main
###############################################################################


def main() -> None:
    """Test function demonstrating line layout with pi digits."""
    page_module: Any = importlib.import_module("ave.page")
    AvSvgPage = page_module.AvSvgPage  # pylint: disable=invalid-name

    # Create SVG page
    viewbox_width: float = 160  # viewbox width in mm
    viewbox_height: float = 15  # viewbox height in mm
    vb_scale: float = 1.0 / viewbox_width  # scale viewbox so that x-coordinates are between 0 and 1
    font_size: float = vb_scale * 3.0  # in mm (already in viewbox units)

    # Create the SVG page
    svg_page: Any = AvSvgPage.create_page_a4(viewbox_width, viewbox_height, vb_scale)

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

    # Define layout parameters
    line_spacing_mm: float = 3.0  # Line spacing in mm
    line_spacing: float = line_spacing_mm * vb_scale  # Convert to normalized coordinates
    x_start: float = 0.0  # Start at left edge of viewbox
    x_end: float = 1.0  # End at right edge of viewbox

    # Load font factory from cache
    # Option 1: Cache-only (works if all needed characters are in cache)
    font_factory: AvGlyphFactory = AvGlyphFactory.load_from_file(
        "fonts/cache/NotoSerif-VariableFont_AA_07_wght0300.json.zip"
    )

    # Option 2: With TTFont fallback for uncached characters (uncomment if needed)
    # ttfont = TTFont("fonts/NotoSerif[wdth,wght].ttf")
    # font_factory = AvGlyphFactory.load_from_file(
    #     "fonts/cache/NotoSerif-VariableFont_AA_07_wght0300.json.zip",
    #     source=TTFontGlyphSource(ttfont)
    # )
    units_per_em: int = font_factory.get_font_properties().units_per_em
    letter_scale: float = font_size / units_per_em

    # Create letter factory
    letter_factory: AvSingleGlyphLetterFactory = AvSingleGlyphLetterFactory(font_factory)

    # Calculate how many lines fit in the viewbox
    viewbox_height_normalized: float = viewbox_height * vb_scale

    # Create sample letter to get font metrics
    sample_letter: AvLetter = letter_factory.create_letter("0", letter_scale, 0, 0)
    sample_bbox: Any = sample_letter.bounding_box

    # Calculate first baseline for top-left corner placement
    # In this coordinate system: y=0 is BOTTOM, y increases UPWARD
    # Top of viewbox is at y = viewbox_height_normalized
    # Letters extend above baseline, top is at baseline + sample_bbox.ymax
    # We want: baseline + sample_bbox.ymax = viewbox_height_normalized
    # So: baseline = viewbox_height_normalized - sample_bbox.ymax
    first_baseline: float = viewbox_height_normalized - sample_bbox.ymax

    # Generate pi digits starting from "3.14..."
    pi_gen: Iterator[str] = GeomMath.pi_digit_generator()
    all_letters: List[AvLetter] = []
    current_baseline: float = first_baseline
    line_count: int = 0
    chars_used: int = 0

    # Fill viewbox with lines from top to bottom
    while current_baseline + sample_bbox.ymin >= 0:  # Bottom of line should not go below y=0
        line_count += 1
        print(f"Working on line {line_count} at baseline y={current_baseline:.4f}...")

        # Generate enough digits for this line
        pi_text: str = ""
        for _ in range(200):  # More than enough for one line
            try:
                pi_text += next(pi_gen)
            except StopIteration:
                break

        if not pi_text:
            break

        # Create character stream from pi digits for this line
        char_stream: AvCharacterStream = AvCharacterStream(pi_text)

        # Create layouter and layout the line
        # layouter = AvTightCharLineLayouter(
        layouter: AvCharLineLayouter = AvCharLineLayouter(
            x_start=x_start,
            x_end=x_end,
            y_baseline=current_baseline,
            stream=char_stream,
            letter_factory=letter_factory,
            scale=letter_scale,
            # margin=0.0,  # 0.1 * vb_scale,
        )

        letters: List[AvLetter] = layouter.layout_line()

        if not letters:
            break

        all_letters.extend(letters)
        chars_used += len(letters)
        print(f"  Placed {len(letters)} characters on line {line_count}")

        # Move to next line (going down)
        current_baseline -= line_spacing

    # Add all letters to SVG page
    for letter in all_letters:
        svg_path: Any = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
        # svg_path: Any = svg_page.drawing.path(
        #     letter.svg_path_string_debug_polyline(stroke_width=0.001), fill="black", stroke="none"
        # )
        svg_page.add(svg_path)

    # Save the page
    svg_filename: str = "data/output/example/svg/layout/pi_line_layout.svgz"
    svg_page.save_as(svg_filename, include_debug_layer=True, pretty=True, compressed=True)
    print(f"Saved pi line layout to: {svg_filename}")
    print(f"Placed {len(all_letters)} characters across {line_count} lines")


if __name__ == "__main__":
    main()
