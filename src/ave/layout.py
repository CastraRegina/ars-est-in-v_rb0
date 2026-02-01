"""Line layout functionality for characters and syllables."""

from __future__ import annotations

import importlib
from typing import List, Optional

from ave.common import Align
from ave.glyph import AvGlyphFactory, AvGlyphPersistentFactory
from ave.letter import AvLetter, AvLetterFactory, AvSingleGlyphLetterFactory
from ave.text import AvStreamBase

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
        letter_factory: Optional[AvLetterFactory] = None,
        glyph_factory: Optional[AvGlyphFactory] = None,
        scale: float = 1.0,
    ) -> None:
        """
        Initialize the line layouter.

        Args:
            x_start: Starting x position for left alignment
            x_end: Ending x position for right alignment
            y_baseline: Y position of the baseline (constant for all letters)
            stream: Stream providing items (characters/syllables) to layout
            letter_factory: Optional factory function to create letters from items.
                If None, uses default factory with glyph_factory.
            glyph_factory: Factory to create glyphs (required if letter_factory is None)
            scale: Scale factor for letter dimensions (default: 1.0)
        """
        self._x_start = x_start
        self._x_end = x_end
        self._y_baseline = y_baseline
        self._stream = stream
        self._scale = scale
        self._letters: List[AvLetter] = []

        if letter_factory is None:
            if glyph_factory is None:
                raise ValueError("Either letter_factory or glyph_factory must be provided")
            self._letter_factory = AvSingleGlyphLetterFactory(glyph_factory)
        else:
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

        # Adjust last letter for right alignment
        if len(self._letters) > 0:
            last_letter = self._letters[-1]
            last_letter.align = Align.RIGHT
            last_letter_width = last_letter.width()
            last_letter.xpos = self._x_end - last_letter_width

        # Distribute excess space among inner letters
        if len(self._letters) > 2:
            # Calculate total width of placed letters
            total_letter_width = sum(letter.width() for letter in self._letters)

            # Calculate excess space
            available_width = self._x_end - self._x_start
            excess_space = available_width - total_letter_width

            if excess_space > 0:
                # Distribute space among inner letters (not first and last)
                num_inner_letters = len(self._letters) - 2
                space_per_gap = excess_space / (num_inner_letters + 1)

                # Adjust positions
                current_x = self._x_start

                # First letter stays at x_start with LEFT alignment
                self._letters[0].align = Align.LEFT
                self._letters[0].xpos = current_x
                current_x += self._letters[0].width() + space_per_gap

                # Inner letters
                for i in range(1, len(self._letters) - 1):
                    self._letters[i].xpos = current_x
                    current_x += self._letters[i].width() + space_per_gap

                # Last letter already positioned at x_end
        elif len(self._letters) == 2:
            # Only two letters, just ensure proper positioning
            self._letters[0].align = Align.LEFT
            self._letters[0].xpos = self._x_start
            # Last letter already positioned at x_end
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
        letter_factory: Optional[AvLetterFactory] = None,
        glyph_factory: Optional[AvGlyphFactory] = None,
        scale: float = 1.0,
    ) -> None:
        """
        Initialize the syllable line layouter.

        Args:
            x_start: Starting x position for left alignment
            x_end: Ending x position for right alignment
            y_baseline: Y position of the baseline (constant for all letters)
            stream: Stream providing syllables to layout (typically AvSyllableStream)
            letter_factory: Optional factory function to create letters from syllables.
                If None, uses default factory with glyph_factory.
            glyph_factory: Factory to create glyphs (required if letter_factory is None)
            scale: Scale factor for letter dimensions (default: 1.0)
        """
        super().__init__(
            x_start=x_start,
            x_end=x_end,
            y_baseline=y_baseline,
            stream=stream,
            letter_factory=letter_factory,
            glyph_factory=glyph_factory,
            scale=scale,
        )


###############################################################################
# main
###############################################################################


def main():
    """Test function demonstrating line layout with pi digits."""
    from ave.geom import GeomMath  # pylint: disable=import-outside-toplevel
    from ave.text import AvCharacterStream  # pylint: disable=import-outside-toplevel

    page_module = importlib.import_module("ave.page")
    AvSvgPage = page_module.AvSvgPage  # pylint: disable=invalid-name

    # Create SVG page
    viewbox_width = 160  # viewbox width in mm
    viewbox_height = 160  # viewbox height in mm
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

    # Load font
    font_factory = AvGlyphPersistentFactory.load_from_file(
        "fonts/cache/RobotoFlex-VariableFont_AA_01_wght0100.json.zip"
    )
    units_per_em = font_factory.get_font_properties().units_per_em
    scale = font_size / units_per_em

    # Create letter factory
    letter_factory = AvSingleGlyphLetterFactory(font_factory)

    # Calculate how many lines fit in the viewbox
    viewbox_height_normalized = viewbox_height * vb_scale

    # Create sample letter to get font metrics
    sample_letter = letter_factory.create_letter("0", scale, 0, 0)
    sample_bbox = sample_letter.bounding_box()

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
        layouter = AvCharLineLayouter(
            x_start=x_start,
            x_end=x_end,
            y_baseline=current_baseline,
            stream=char_stream,
            letter_factory=letter_factory,
            scale=scale,
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
