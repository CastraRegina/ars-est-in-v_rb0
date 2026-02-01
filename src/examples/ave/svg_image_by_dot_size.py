"""Multi-glyph container that extends AvGlyph for managing collections of AvGlyph objects."""

from __future__ import annotations

from dataclasses import dataclass

from ave.image import AvImage
from ave.page import AvSvgPage
from ave.path_processing import AvPathCreator


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

        # Calculate spacing to fit exactly circles_per_row horizontally
        spacing_x = 1.0 / circles_per_row

        # Use the same spacing in y direction for absolute uniform spacing
        spacing_y = spacing_x

        # Calculate how many circles fit vertically with this spacing
        viewport_height = viewbox_scale * viewbox_height_mm
        circles_per_col = int(viewport_height / spacing_y)

        # Add circles to the SVG page
        for row in range(circles_per_col):
            for col in range(circles_per_row):
                # Calculate position - centered in each grid cell
                x = col * spacing_x + spacing_x / 2
                y = row * spacing_y + spacing_y / 2

                # Get the normalized gray value at this position (0.0=black, 1.0=white)
                # Use a small region around the point for sampling
                region_size = min(spacing_x, spacing_y) * 0.5
                gray_normalized = self._av_image.get_region_weighted_mean_normalized_rel(
                    x - region_size / 2, y - region_size / 2, x + region_size / 2, y + region_size / 2
                )

                # Map normalized gray value to circle area, then calculate diameter:
                # black (0.0) -> 99% of spacing diameter (large dots, heavy visual weight)
                # white (1.0) -> 5% of spacing diameter (small dots, light visual weight)
                # Since we want diameter range of 5% to 99%, we need to work backwards:
                # Area at 99% diameter = (0.99)^2 = 0.9801 or 98.01%
                # Area at 5% diameter = (0.05)^2 = 0.0025 or 0.25%
                # Formula: area_factor = 0.9801 - gray_normalized * (0.9801 - 0.0025)
                # This directly maps: black (0.0) -> large (0.9801), white (1.0) -> small (0.0025)
                area_factor = 0.9801 - gray_normalized * 0.9776
                diameter_factor = area_factor**0.5  # Square root for area-to-diameter conversion
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

    # Process multiple example files
    input_files = [
        (
            "data/output/example/png/test_board_07x07_10grays.png",
            "data/output/example/svg/dots/test_board_07x07_10grays.svgz",
        ),
        (
            "data/output/example/png/test_board_09x09_10grays.png",
            "data/output/example/svg/dots/test_board_09x09_10grays.svgz",
        ),
        (
            "data/input/example/pics/MC.jpg",
            "data/output/example/svg/dots/MC.svgz",
        ),
        (
            "data/input/example/pics/BT.png",
            "data/output/example/svg/dots/BT.svgz",
        ),
    ]

    for image_path, output_svg in input_files:
        print(f"\nProcessing: {image_path}")
        converter = ImageToSvgDotConverter(image_path)
        converter.svg_page().save_as(output_svg, compressed=True)
        print(f"SVG saved to: {output_svg}")


if __name__ == "__main__":
    main()
