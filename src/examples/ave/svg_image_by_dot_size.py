"""Multi-glyph container that extends AvGlyph for managing collections of AvGlyph objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ave.glyph import AvGlyph
from ave.image import AvImage
from ave.page import AvSvgPage
from ave.path_processing import AvPathCreator

# @dataclass
# class AvMultiGlyph(AvGlyph):
#     """Container for multiple AvGlyph objects that extends AvGlyph.

#     Represents a collection of glyphs that can be managed together,
#     such as a font subset, ligature components, or decorative elements.
#     Inherits from AvGlyph to provide compatibility with existing glyph systems.
#     """

#     _glyphs: List[AvGlyph] = field(default_factory=list)

#     def __init__(self, glyphs: Optional[List[AvGlyph]] = None):
#         """Initialize AvMultiGlyph with list of glyphs.

#         Args:
#             glyphs: List of AvGlyph objects to contain
#         """
#         # Initialize parent AvGlyph with placeholder values
#         super().__init__(character="", width=0.0, path=None)

#         # Set and validate glyphs
#         self._glyphs = glyphs or []

#         for i, glyph in enumerate(self._glyphs):
#             if not isinstance(glyph, AvGlyph):
#                 raise TypeError(f"Glyph at index {i} is not an AvGlyph object")

#     @property
#     def glyphs(self) -> List[AvGlyph]:
#         """Get the glyphs list."""
#         return self._glyphs

#     @property
#     def character(self) -> str:
#         """Get combined characters of all glyphs."""
#         return "".join(glyph.character for glyph in self._glyphs)

#     def width(self) -> float:
#         """Calculate total width of all glyphs."""
#         return sum(glyph.width() for glyph in self._glyphs)

#     def path(self):
#         """Get combined path of all glyphs, positioned horizontally."""
#         if not self._glyphs:
#             return None

#         # Import here to avoid circular imports
#         from ave.path import AvPath

#         if len(self._glyphs) == 1:
#             return self._glyphs[0].path()

#         # Combine paths by positioning glyphs horizontally
#         combined_paths = []
#         current_x = 0.0

#         for glyph in self._glyphs:
#             glyph_path = glyph.path()
#             if glyph_path is None:
#                 continue

#             # Transform path to current position
#             transformed_path = glyph_path.transform([1.0, 0.0, 0.0, 1.0, current_x, 0.0])
#             combined_paths.append(transformed_path)
#             current_x += glyph.width()

#         # Merge all paths
#         if combined_paths:
#             return AvPath.merge_paths(combined_paths)
#         return None

#     def bounding_box(self):
#         """Get bounding box that encompasses all glyphs."""
#         if not self._glyphs:
#             from ave.geom import AvBox

#             return AvBox(0.0, 0.0, 0.0, 0.0)

#         if len(self._glyphs) == 1:
#             return self._glyphs[0].bounding_box()

#         # Calculate combined bounding box
#         current_x = 0.0
#         min_x, min_y = float("inf"), float("inf")
#         max_x, max_y = float("-inf"), float("-inf")

#         for glyph in self._glyphs:
#             bbox = glyph.bounding_box()
#             # Transform bbox to current position
#             glyph_min_x = bbox.xmin + current_x
#             glyph_max_x = bbox.xmax + current_x
#             glyph_min_y = bbox.ymin
#             glyph_max_y = bbox.ymax

#             min_x = min(min_x, glyph_min_x)
#             max_x = max(max_x, glyph_max_x)
#             min_y = min(min_y, glyph_min_y)
#             max_y = max(max_y, glyph_max_y)

#             current_x += glyph.width()

#         from ave.geom import AvBox

#         return AvBox(min_x, min_y, max_x, max_y)

#     def left_side_bearing(self) -> float:
#         """Get left side bearing of the first glyph."""
#         if not self._glyphs:
#             return 0.0
#         return self._glyphs[0].left_side_bearing()

#     def right_side_bearing(self) -> float:
#         """Get right side bearing of the last glyph."""
#         if not self._glyphs:
#             return 0.0
#         return self._glyphs[-1].right_side_bearing()

#     def approx_equal(self, other: AvGlyph, rtol: float = 1e-9, atol: float = 1e-9) -> bool:
#         """Check if multi-glyph equals another glyph within tolerances."""
#         if not isinstance(other, AvMultiGlyph):
#             return False

#         if len(self._glyphs) != len(other._glyphs):
#             return False

#         for i, (self_glyph, other_glyph) in enumerate(zip(self._glyphs, other._glyphs)):
#             if not self_glyph.approx_equal(other_glyph, rtol, atol):
#                 return False

#         return True

#     def revise_direction(self) -> AvGlyph:
#         """Revise direction for all contained glyphs."""
#         revised_glyphs = []
#         for glyph in self._glyphs:
#             revised_glyph = glyph.revise_direction()
#             revised_glyphs.append(revised_glyph)

#         return AvMultiGlyph(revised_glyphs)

#     # Additional multi-glyph specific methods

#     def add_glyph(self, glyph: AvGlyph) -> None:
#         """Add a glyph to the collection."""
#         if not isinstance(glyph, AvGlyph):
#             raise TypeError("glyph must be an AvGlyph object")
#         self._glyphs.append(glyph)

#     def remove_glyph(self, index: int) -> AvGlyph:
#         """Remove a glyph by index."""
#         if index < 0 or index >= len(self._glyphs):
#             raise IndexError(f"Index {index} out of range for {len(self._glyphs)} glyphs")
#         return self._glyphs.pop(index)

#     def get_glyph(self, index: int) -> AvGlyph:
#         """Get a glyph by index."""
#         if index < 0 or index >= len(self._glyphs):
#             raise IndexError(f"Index {index} out of range for {len(self._glyphs)} glyphs")
#         return self._glyphs[index]

#     def find_by_character(self, character: str) -> Optional[AvGlyph]:
#         """Find a glyph by its character."""
#         for glyph in self._glyphs:
#             if glyph.character == character:
#                 return glyph
#         return None

#     def get_characters(self) -> List[str]:
#         """Get list of all characters."""
#         return [glyph.character for glyph in self._glyphs]

#     def count(self) -> int:
#         """Get number of glyphs."""
#         return len(self._glyphs)

#     def is_empty(self) -> bool:
#         """Check if collection is empty."""
#         return len(self._glyphs) == 0

#     def clear(self) -> None:
#         """Remove all glyphs."""
#         self._glyphs.clear()

#     def __len__(self) -> int:
#         """Get number of glyphs using len()."""
#         return len(self._glyphs)

#     def __iter__(self):
#         """Iterate over glyphs."""
#         return iter(self._glyphs)

#     def __getitem__(self, index: int) -> AvGlyph:
#         """Get glyph using indexing syntax."""
#         return self.get_glyph(index)

#     def __repr__(self) -> str:
#         """String representation."""
#         chars = ", ".join(f"'{g.character}'" for g in self._glyphs[:5])
#         if len(self._glyphs) > 5:
#             chars += f", ... ({len(self._glyphs) - 5} more)"
#         return f"AvMultiGlyph([{chars}])"


class AvCircleGlyph(AvGlyph):
    """A glyph representing a circle shape.

    Specialized glyph that creates a circular path using the AvPathCreator.
    The circle is centered within the glyph's width and height.
    """

    def __init__(self, width: float, diameter: float):
        """Initialize a circle glyph.

        Args:
            width (float): The width of the glyph (also used as height)
            diameter (float): The diameter of the circle to create

        Note:
            - The circle is centered at (width/2, width/2)
            - Uses AvPathCreator.circle() with 4 quadratic Bezier curves
            - Character is empty string as this is a decorative glyph
        """
        # Create circular path centered in the glyph space
        circle_path = AvPathCreator.circle(width / 2, width / 2, diameter)

        # Initialize parent AvGlyph
        super().__init__(character="", width=width, path=circle_path)


@dataclass
class ImageToSvgDotConverter:
    """Converts a PNG image to SVG with proper scaling and viewBox setup.

    Loads a PNG image into an AvImage and creates a corresponding AvSvgPage
    with a viewBox of 160mm width. The height is calculated based on the
    image aspect ratio. Draws a rectangle matching the image dimensions.
    """

    _av_image: AvImage = None
    _svg_page: AvSvgPage = None

    def __init__(self, image_path: str):
        """Initialize the converter with an image path.

        Args:
            image_path (str): Path to the PNG image file
        """
        self._load_image(image_path)
        self._create_svg_page()

    def _load_image(self, image_path: str) -> None:
        """Load the PNG image into an AvImage."""
        self._av_image = AvImage.from_file(image_path)

    def _create_svg_page(self) -> None:
        """Create an SVG page with proper viewBox and scaling."""
        # Get image dimensions
        img_width = self._av_image.width_px
        img_height = self._av_image.height_px

        # Calculate viewBox: 160mm width, scaled height
        # 160mm = 160 units in SVG (assuming 1 unit = 1mm)
        viewbox_width_mm = 160.0
        aspect_ratio = img_height / img_width
        viewbox_height_mm = viewbox_width_mm * aspect_ratio

        # Calculate scale to make width = 1.0
        viewbox_scale = 1.0 / viewbox_width_mm

        # Create SVG page using A4 template with centered viewBox
        self._svg_page = AvSvgPage.create_page_a4(
            viewbox_width_mm=viewbox_width_mm,
            viewbox_height_mm=viewbox_height_mm,
            viewbox_scale=viewbox_scale,
        )

        # Create a rectangle matching the image dimensions using AvPathCreator
        # Use scaled coordinates (width = 1.0, height = scaled)
        rect_path = AvPathCreator.rectangle(x1=0.0, y1=0.0, x2=1.0, y2=viewbox_scale * viewbox_height_mm)

        # Convert the AvPath to SVG path string
        svg_path = self._svg_page.drawing.path(
            d=rect_path.svg_path_string(),
            fill="none",
            stroke="black",
            stroke_width=0.1 * viewbox_scale,
        )
        self._svg_page.add(svg_path)

        # Define circles per row to the SVG page
        circles_per_row = 100

        # Calculate spacing to fit exactly 50 circles horizontally
        spacing_x = 1.0 / circles_per_row

        # Use the same spacing in y direction for absolute uniform spacing
        spacing_y = spacing_x

        # Calculate how many circles fit vertically with this spacing
        viewport_height = viewbox_scale * viewbox_height_mm
        circles_per_col = int(viewport_height / spacing_y)

        # Add circles to the SVG page - exactly 50x50 grid with varying sizes
        for row in range(circles_per_col):
            for col in range(circles_per_row):
                # Calculate position - centered in each grid cell
                x = col * spacing_x + spacing_x / 2
                y = row * spacing_y + spacing_y / 2

                # Get the gray value at this position (0-255)
                # Use a small region around the point for sampling
                region_size = min(spacing_x, spacing_y) * 0.5
                gray_value = self._av_image.get_region_weighted_mean_rel(
                    x - region_size / 2, y - region_size / 2, x + region_size / 2, y + region_size / 2
                )

                # Map gray value to circle diameter:
                # black (0) -> 90% of spacing, white (255) -> 10% of spacing
                # Formula: diameter = 0.9 - (gray_value / 255) * 0.8
                diameter_factor = 0.9 - (gray_value / 255.0) * 0.8
                current_diameter = min(spacing_x, spacing_y) * diameter_factor

                # Create a circle at this position with varying diameter
                circle_path = AvPathCreator.circle(cx=x, cy=y, diameter=current_diameter)

                # Add the circle to the SVG page - filled with black
                svg_circle = self._svg_page.drawing.path(
                    d=circle_path.svg_path_string(),
                    fill="black",  # Filled circles
                    stroke="none",  # No stroke
                )
                self._svg_page.add(svg_circle)

        print(
            f"Added {circles_per_row * circles_per_col} circles ({circles_per_row}x{circles_per_col} grid) to the SVG"
        )

    def image(self) -> AvImage:
        """Get the loaded AvImage.

        Returns:
            AvImage: The loaded image
        """
        return self._av_image

    def svg_page(self) -> AvSvgPage:
        """Get the created SVG page.

        Returns:
            AvSvgPage: The SVG page with rectangle
        """
        return self._svg_page


def main() -> None:
    """
    Main function for the ImageToSvgDotConverter example.
    """

    # Demonstrate ImageToSvgConverter
    print("\n" + "=" * 50)
    print("Image to SVG Conversion:")
    print("=" * 50)

    # Create converter for the test board image
    image_path = "data/output/example/png/test_board_09x09_10grays.png"
    converter = ImageToSvgDotConverter(image_path)

    # Save the SVG as .svgz (zipped SVG)
    output_svg = "data/output/example/svg/test_board_09x09_10grays.svgz"
    converter.svg_page().save_as(output_svg, compressed=True)
    print(f"\nSVG saved to: {output_svg}")


if __name__ == "__main__":
    main()
