"""Line layout functionality for characters and syllables."""

from __future__ import annotations

import importlib
from typing import List

from ave.common import Align
from ave.geom import GeomMath
from ave.glyph_factory import AvGlyphFactory
from ave.letter import AvLetter, AvLetterFactory, AvSingleGlyphLetterFactory
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

        Process:
            1. Place letters sequentially until x_end is reached
            2. If last letter doesn't fit, give it back to stream
            3. Distribute excess space among letters (except first and last)
        """
        # Reset letters list
        self._letters = []

        # First pass: place letters and check fit
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
                    None,  # No alignment initially
                )

                # Set left neighbor if not first letter
                if self._letters:
                    letter.left_letter = self._letters[-1]

                # Check if letter fits
                letter_width = letter.width()
                letter_right_edge = current_x + letter_width

                if letter_right_edge > self._x_end and self._letters:
                    # Letter doesn't fit, give it back to stream
                    self._stream.rewind(1)
                    break

                # Letter fits, add to list
                self._letters.append(letter)
                current_x += letter_width

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
            first_letter = self._letters[0]
            first_letter.align = Align.LEFT
            first_letter.xpos = self._x_start

        # Adjust last letter for right alignment
        if len(self._letters) > 1:
            last_letter = self._letters[-1]
            last_letter.align = Align.RIGHT
            last_letter_width = last_letter.width()
            last_letter.xpos = self._x_end - last_letter_width

        # Distribute excess space by shifting inner letters proportionally so
        # that each gap (including the first and the one before the last
        # letter) grows by the same amount.
        if len(self._letters) > 2:
            # Calculate excess: difference between right-aligned position
            # and the original tight position of the last letter
            excess = self._letters[-1].xpos - initial_positions[-1]
            if excess > 0:
                denom = len(self._letters) - 1
                for idx in range(1, len(self._letters) - 1):
                    shift = excess * idx / denom
                    self._letters[idx].xpos = initial_positions[idx] + shift

        elif len(self._letters) == 2:
            # Last letter already positioned at x_end
            pass
        elif len(self._letters) == 1:
            # Single letter, center it
            single_letter = self._letters[0]
            single_letter_width = single_letter.width()
            center_offset = (self._x_end - self._x_start - single_letter_width) / 2
            single_letter.xpos = self._x_start + center_offset

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
    sequentially and distributes excess space among them.
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
                    None,  # No alignment initially
                )

                # Ensure first letter uses LEFT alignment immediately so that
                # spacing calculations reference the correct geometry.
                if not self._letters:
                    letter.align = Align.LEFT
                    letter.xpos = self._x_start

                # Apply tight packing if there is a left neighbor
                if self._letters:
                    letter.left_letter = self._letters[-1]

                    # Calculate space to left neighbor and move letter to touch
                    space = letter.left_space()
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

        # Ensure first letter is left-aligned at x_start
        if len(self._letters) > 0:
            first_letter = self._letters[0]
            first_letter.align = Align.LEFT
            first_letter.xpos = self._x_start

        # Save last letter position before adjusting for right alignment
        tight_last_xpos = self._letters[-1].xpos if len(self._letters) > 1 else None

        # Adjust last letter for right alignment
        if len(self._letters) > 1:
            last_letter = self._letters[-1]
            last_letter.align = Align.RIGHT
            last_letter.xpos = self._x_end - last_letter.width()

        # Distribute excess space by shifting inner letters proportionally so
        # that each gap (including the first and the one before the last
        # letter) grows by the same amount.
        if len(self._letters) > 2 and tight_last_xpos is not None:
            excess = self._letters[-1].xpos - tight_last_xpos
            if excess > 0:
                denom = len(self._letters) - 1
                for idx in range(1, len(self._letters) - 1):
                    shift = excess * idx / denom
                    self._letters[idx].xpos += shift

        elif len(self._letters) == 1:
            # Single letter, center it
            single_letter = self._letters[0]
            center_offset = (self._x_end - self._x_start - single_letter.width()) / 2
            single_letter.xpos = self._x_start + center_offset

        return self._letters


###############################################################################
# main
###############################################################################


def main():
    """Test function demonstrating line layout with pi digits."""
    page_module = importlib.import_module("ave.page")
    AvSvgPage = page_module.AvSvgPage  # pylint: disable=invalid-name

    # Create SVG page
    viewbox_width = 160  # viewbox width in mm
    viewbox_height = 15  # viewbox height in mm
    vb_scale = 1.0 / viewbox_width  # scale viewbox so that x-coordinates are between 0 and 1
    font_size = vb_scale * 3.0  # in mm (already in viewbox units)

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

    # Define layout parameters
    line_spacing_mm = 3.0  # Line spacing in mm
    line_spacing = line_spacing_mm * vb_scale  # Convert to normalized coordinates
    x_start = 0.0  # Start at left edge of viewbox
    x_end = 1.0  # End at right edge of viewbox

    # Load font factory from cache
    # Option 1: Cache-only (works if all needed characters are in cache)
    font_factory = AvGlyphFactory.load_from_file("fonts/cache/NotoSerif-VariableFont_AA_07_wght0300.json.zip")

    # Option 2: With TTFont fallback for uncached characters (uncomment if needed)
    # ttfont = TTFont("fonts/NotoSerif[wdth,wght].ttf")
    # font_factory = AvGlyphFactory.load_from_file(
    #     "fonts/cache/NotoSerif-VariableFont_AA_07_wght0300.json.zip",
    #     source=TTFontGlyphSource(ttfont)
    # )
    units_per_em = font_factory.get_font_properties().units_per_em
    letter_scale = font_size / units_per_em

    # Create letter factory
    letter_factory = AvSingleGlyphLetterFactory(font_factory)

    # Calculate how many lines fit in the viewbox
    viewbox_height_normalized = viewbox_height * vb_scale

    # Create sample letter to get font metrics
    sample_letter = letter_factory.create_letter("0", letter_scale, 0, 0)
    sample_bbox = sample_letter.bounding_box

    # Calculate first baseline for top-left corner placement
    # In this coordinate system: y=0 is BOTTOM, y increases UPWARD
    # Top of viewbox is at y = viewbox_height_normalized
    # Letters extend above baseline, top is at baseline + sample_bbox.ymax
    # We want: baseline + sample_bbox.ymax = viewbox_height_normalized
    # So: baseline = viewbox_height_normalized - sample_bbox.ymax
    first_baseline = viewbox_height_normalized - sample_bbox.ymax

    # Generate pi digits starting from "3.14..."
    pi_gen = GeomMath.pi_digit_generator()
    all_letters = []
    current_baseline = first_baseline
    line_count = 0
    chars_used = 0

    # Fill viewbox with lines from top to bottom
    while current_baseline + sample_bbox.ymin >= 0:  # Bottom of line should not go below y=0
        line_count += 1
        print(f"Working on line {line_count} at baseline y={current_baseline:.4f}...")

        # Generate enough digits for this line
        pi_text = ""
        for _ in range(200):  # More than enough for one line
            try:
                pi_text += next(pi_gen)
            except StopIteration:
                break

        if not pi_text:
            break

        # Create character stream from pi digits for this line
        char_stream = AvCharacterStream(pi_text)

        # Create layouter and layout the line
        layouter = AvTightCharLineLayouter(
            # layouter = AvCharLineLayouter(
            x_start=x_start,
            x_end=x_end,
            y_baseline=current_baseline,
            stream=char_stream,
            letter_factory=letter_factory,
            scale=letter_scale,
            margin=0.0,  # 0.1 * vb_scale,
        )

        letters = layouter.layout_line()

        if not letters:
            break

        all_letters.extend(letters)
        chars_used += len(letters)
        print(f"  Placed {len(letters)} characters on line {line_count}")

        # Move to next line (going down)
        current_baseline -= line_spacing

    # Add all letters to SVG page
    for letter in all_letters:
        svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
        svg_page.add(svg_path)

    # Save the page
    svg_filename = "data/output/example/svg/layout/pi_line_layout.svgz"
    svg_page.save_as(svg_filename, include_debug_layer=True, pretty=True, compressed=True)
    print(f"Saved pi line layout to: {svg_filename}")
    print(f"Placed {len(all_letters)} characters across {line_count} lines")


if __name__ == "__main__":
    main()
