"""Multi-weight letter rendering example using cached font files."""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import List

from ave.common import AlignX, CenterRef
from ave.glyph_factory import AvGlyphFactory
from ave.letter import AvMultiWeightLetter
from ave.letter_processing import AvLetterAlignment
from ave.page import AvSvgPage


def discover_font_basenames(path_name: str) -> List[str]:
    """
    Discover all unique font basenames in the cache directory.

    This groups font files by their base name (everything before the _XX weight suffix).
    For example, "Grandstander-VariableFont_AA_01_wght0100.json.zip" and
    "Grandstander-VariableFont_AA_02_wght0200.json.zip" both belong to
    "Grandstander-VariableFont_AA_" base font.

    Args:
        path_name: Directory path where cached files are stored (e.g., "fonts/cache")

    Returns:
        List[str]: List of unique font basenames
    """
    cache_dir = Path(path_name)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {path_name}")

    # Find all font files
    pattern = "*_wght*.json.zip"
    file_paths = list(cache_dir.glob(pattern))

    if not file_paths:
        raise FileNotFoundError(f"No cached font files found with pattern: {pattern}")

    # Extract unique basenames (everything before _XX_wght)
    basenames = set()
    for file_path in file_paths:
        # Extract basename before _XX_wght (where XX is the weight number)
        # Pattern: FontName_XX_wghtYYYY.json.zip
        match = re.match(r"(.+?)_\d+_wght\d+", file_path.name)
        if match:
            basenames.add(match.group(1) + "_")

    # Return sorted list
    return sorted(list(basenames))


def load_cached_fonts(path_name: str, font_fn_base: str) -> List[AvGlyphFactory]:
    """
    Load cached font files from directory.

    Args:
        path_name: Directory path where cached files are stored (e.g., "fonts/cache")
        font_fn_base: Base filename pattern (e.g., "RobotoFlex-VariableFont_AA_")

    Returns:
        List[AvGlyphFactory]: Ordered list of font factories (lightest to heaviest)
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
            # Load cache-only factory (works for all cached characters)
            factory = AvGlyphFactory.load_from_file(str(file_path))
            factories.append(factory)
            print(f"Loaded: {file_path.name}")
        except (FileNotFoundError, ValueError, RuntimeError, OSError, gzip.BadGzipFile) as e:
            print(f"Warning: Failed to load {file_path.name}: {e}")

    if not factories:
        raise ValueError("No valid cached font files could be loaded")

    return factories


def render_letter_with_boxes(svg_page, multi_letter, colors, stroke_width):
    """Render a multi-weight letter with bounding boxes."""
    # Render all weight variants (already in heavy-to-light order)
    for letter, color in zip(multi_letter.letters, colors):
        svg_path = svg_page.drawing.path(letter.svg_path_string(), fill=color, stroke="none")
        svg_page.add(svg_path)

    # Add bounding box in yellow
    bbox = multi_letter.bounding_box
    svg_bbox = svg_page.drawing.path(
        f"M {bbox.xmin:g} {bbox.ymin:g} "
        f"L {bbox.xmax:g} {bbox.ymin:g} "
        f"L {bbox.xmax:g} {bbox.ymax:g} "
        f"L {bbox.xmin:g} {bbox.ymax:g} Z",
        fill="none",
        stroke="yellow",
        stroke_width=stroke_width,
    )
    svg_page.add(svg_bbox, True)

    # Add letter box in green
    lbox = multi_letter.letter_box
    svg_lbox = svg_page.drawing.path(
        f"M {lbox.xmin:g} {lbox.ymin:g} "
        f"L {lbox.xmax:g} {lbox.ymin:g} "
        f"L {lbox.xmax:g} {lbox.ymax:g} "
        f"L {lbox.xmin:g} {lbox.ymax:g} Z",
        fill="none",
        stroke="green",
        stroke_width=stroke_width,
    )
    svg_page.add(svg_lbox, True)


def render_alignment_test(svg_page, one_font_factories, scale, font_size, test_ypos, colors, stroke_width):
    """Render alignment test letters (U and A) at both edges."""
    # U at x-pos=0 with LEFT alignment
    multi_letter_u_left = AvMultiWeightLetter.from_factories(
        character="U",
        factories=one_font_factories,
        scale=scale,
        xpos=0.0,
        ypos=test_ypos,
    )
    AvLetterAlignment.align_to_x_border(multi_letter_u_left, 0.0, AlignX.LEFT)
    render_letter_with_boxes(svg_page, multi_letter_u_left, colors, stroke_width)

    # U at x-pos=1.0 with RIGHT alignment
    multi_letter_u_right = AvMultiWeightLetter.from_factories(
        character="U",
        factories=one_font_factories,
        scale=scale,
        xpos=1.0,
        ypos=test_ypos,
    )
    AvLetterAlignment.align_to_x_border(multi_letter_u_right, 1.0, AlignX.RIGHT)
    render_letter_with_boxes(svg_page, multi_letter_u_right, colors, stroke_width)

    # A at x-pos=0 with LEFT alignment (one font-size above)
    multi_letter_a_left = AvMultiWeightLetter.from_factories(
        character="A",
        factories=one_font_factories,
        scale=scale,
        xpos=0.0,
        ypos=test_ypos + font_size,
    )
    AvLetterAlignment.align_to_x_border(multi_letter_a_left, 0.0, AlignX.LEFT)
    render_letter_with_boxes(svg_page, multi_letter_a_left, colors, stroke_width)

    # A at x-pos=1.0 with RIGHT alignment (one font-size above)
    multi_letter_a_right = AvMultiWeightLetter.from_factories(
        character="A",
        factories=one_font_factories,
        scale=scale,
        xpos=1.0,
        ypos=test_ypos + font_size,
    )
    AvLetterAlignment.align_to_x_border(multi_letter_a_right, 1.0, AlignX.RIGHT)
    render_letter_with_boxes(svg_page, multi_letter_a_right, colors, stroke_width)


def main():
    """Test function for AvMultiWeightLetter."""

    # Example usage
    try:
        # Colors for multi-weight rendering (heavy to light, black to light gray)
        # Index 0 = heaviest (black), Index N-1 = lightest (light gray)
        colors = [
            "#000000",  # Index 0: Heaviest weight (black)
            "#202020",
            "#404040",
            "#606060",
            "#808080",
            "#A0A0A0",
            "#C0C0C0",
            "#D0D0D0",
            "#E0E0E0",  # Index 8: Lightest weight (light gray)
        ]

        # Discover all available fonts in cache
        font_basenames = discover_font_basenames("fonts/cache")
        print(f"Discovered {len(font_basenames)} fonts:")
        for basename in font_basenames:
            print(f"  - {basename}")

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

        # Render characters with multi-weight for each font
        line_height = 0.04  # Space between font lines
        current_ypos = 0.02

        for font_idx, font_basename in enumerate(font_basenames):
            print(f"Rendering font {font_idx + 1}/{len(font_basenames)}: {font_basename}...", end="", flush=True)

            # Load font for this line
            one_font_factories = load_cached_fonts("fonts/cache", font_basename)

            # Get units_per_em from the current font factory
            units_per_em = one_font_factories[0].get_font_properties().units_per_em if one_font_factories else 2048.0
            scale = font_size / units_per_em  # proper scale calculation for the current font

            # Reset x position for each font line
            current_xpos = 0.05

            # Calculate stroke width based on font dash thickness
            font_props = one_font_factories[0].get_font_properties()
            stroke_width = 0.1 * font_props.dash_thickness * font_size / font_props.units_per_em

            for char in characters:
                # Create multi-weight letter for this character
                multi_letter = AvMultiWeightLetter.from_factories(
                    character=char,
                    factories=one_font_factories,
                    scale=scale,  # Use proper scale, not font_size
                    xpos=current_xpos,
                    ypos=current_ypos,
                )
                # Align multi-weight letter to center
                if multi_letter.letters:
                    center_x, center_y = multi_letter.letters[0].letter_box.center
                    AvLetterAlignment.align_to_center(multi_letter.letters, center_x, center_y, CenterRef.LETTER_BOX)

                # Render each weight variant with different opacity
                render_letter_with_boxes(svg_page, multi_letter, colors, stroke_width)

                # Move to next position
                current_xpos += multi_letter.width() + 0.002

            # Render alignment test letters
            test_ypos = current_ypos  # Same line as the characters
            stroke_width = 0.0001
            render_alignment_test(svg_page, one_font_factories, scale, font_size, test_ypos, colors, stroke_width)

            # Add font name at the end of the line
            font_display_name = font_basename.replace("-VariableFont_AA_", "")
            font_text = f" -- {font_display_name}"

            for char in font_text:
                # Create multi-weight letter for font name character
                multi_letter = AvMultiWeightLetter.from_factories(
                    character=char,
                    factories=one_font_factories,
                    scale=scale,
                    xpos=current_xpos,
                    ypos=current_ypos,
                )
                # Align multi-weight letter to center
                if multi_letter.letters:
                    center_x, center_y = multi_letter.letters[0].letter_box.center
                    AvLetterAlignment.align_to_center(multi_letter.letters, center_x, center_y, CenterRef.LETTER_BOX)

                # Render heaviest weight in black (bottom layer) - index 0 is now heaviest
                svg_path_heavy = svg_page.drawing.path(
                    multi_letter.letters[0].svg_path_string(), fill="#000000", stroke="none"
                )
                svg_page.add(svg_path_heavy)

                # Render lightest weight in light gray (top layer) - index -1 is now lightest
                svg_path_light = svg_page.drawing.path(
                    multi_letter.letters[-1].svg_path_string(), fill="#E0E0E0", stroke="none"
                )
                svg_page.add(svg_path_light)

                # Move to next position
                current_xpos += multi_letter.width() + 0.001

            # Move to next line for next font
            current_ypos += line_height
            print(" done")

        # Save the SVG
        svg_filename = "data/output/example/svg/ave/example_multi_weight_letters.svgz"
        print(f"\nSaving to {svg_filename} ...")
        svg_page.save_as(svg_filename, include_debug_layer=True, pretty=True, compressed=True)
        print(f"Saved to {svg_filename}")

    except (FileNotFoundError, ValueError, RuntimeError, OSError, gzip.BadGzipFile) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
