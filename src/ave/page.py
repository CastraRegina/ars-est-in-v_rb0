"""SVG page representation and layout utilities for document generation."""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from typing import Union

import svgwrite
import svgwrite.base
import svgwrite.container
import svgwrite.elementfactory
from fontTools.ttLib import TTFont
from svgwrite.extensions import Inkscape

from ave.common import Align
from ave.font import AvFont
from ave.fonttools import FontHelper
from ave.glyph import AvGlyphCachedSourceFactory, AvGlyphFromTTFontFactory
from ave.letter import AvSingleGlyphLetter


@dataclass
class AvSvgPage:
    """A page (canvas) described by SVG with a viewbox to draw inside.

    Coordinate System:
        - The canvas uses physical dimensions in millimeters (e.g., A4: 210x297 mm)
        - The viewbox defines a drawable area with its own coordinate system
        - Viewbox coordinates start at (0, 0) in the bottom-left corner
        - X increases from left to right, Y increases from bottom to top
        - Maximum coordinates are (viewbox_width, viewbox_height) in the top-right
        - A scale factor (viewbox_scale) can be applied to normalize coordinates
            (e.g., scale=1/width makes X range from 0 to 1)

    Example: With viewbox_width=170mm, viewbox_height=120mm, scale=1/170:
        - Bottom-left corner: (0, 0)
        - Top-right corner: (1.0, 120/170 ≈ 0.706)
        - Center point: (0.5, 0.353)

    Contains groups/layers:
        - root       -- (group) contains y-flip and translation to bottom-left origin
            - main   -- editable->locked=False  --  hidden->display="block"
            - debug  -- editable->locked=False  --  hidden->display="none"
    """

    _inkscape: Inkscape  # extension to support layers

    drawing: svgwrite.Drawing
    root_group: svgwrite.container.Group
    main_layer: svgwrite.container.Group
    debug_layer: svgwrite.container.Group

    def __init__(
        self,
        canvas_width_mm: float,
        canvas_height_mm: float,
        viewbox_x_mm: float,
        viewbox_y_mm: float,
        _viewbox_width_mm: float,  # not used for the moment
        viewbox_height_mm: float,
        viewbox_scale: float = 1.0,
    ):
        """
        Initialize the SVG page with specified canvas and viewbox dimensions.

        The viewbox is the drawing area with its own coordinate-system left-to-right and bottom-to-top.
        viewbox_scale scales the viewbox coordinates for further use (e.g. by the add()-method).

        Args:
            canvas_width_mm (float): The width of the canvas (=whole page) in millimeters.
            canvas_height_mm (float): The height of the canvas (=whole page) in millimeters.
            viewbox_x_mm (float): The x-coordinate of the viewbox's top-left point in millimeters.
            viewbox_y_mm (float): The y-coordinate of the viewbox's top-left point in millimeters.
            viewbox_width_mm (float): The width of the viewbox in millimeters left-to-right.
            viewbox_height_mm (float): The height of the viewbox in millimeters top-to-bottom.
            viewbox_scale (float, optional): The scale factor for the viewbox. Defaults to 1.0.
        """

        # calculate viewbox coordinates, i.e. the canvas coordinates from viewbox perspective
        vb_x: float = -viewbox_x_mm * viewbox_scale
        vb_y: float = -viewbox_y_mm * viewbox_scale
        vb_width: float = viewbox_scale * canvas_width_mm  # canvas width relativ to viewbox
        vb_height: float = viewbox_scale * canvas_height_mm  # canvas height relativ to viewbox

        # Setup canvas and viewbox. profile="full" to support numbers with more than 4 decimal digits
        self.drawing = svgwrite.Drawing(
            size=(f"{canvas_width_mm}mm", f"{canvas_height_mm}mm"),
            viewBox=(f"{vb_x} {vb_y} {vb_width} {vb_height}"),
            profile="full",
        )

        # Define root group with transformation to flip y-axis and set origin to bottom-left
        y_translate = -viewbox_height_mm * viewbox_scale
        self.root_group = self.drawing.g(id="root", transform=f"scale(1,-1) translate(0,{y_translate})")

        # Initialize Inkscape extension for layer support
        self._inkscape = Inkscape(self.drawing)

        # Define layers
        self.main_layer = self._inkscape.layer(label="main", locked=False)
        self.debug_layer = self._inkscape.layer(label="debug", locked=False, display="none")

        # Assemble the tree structure: drawing -> root_group -> [debug_layer, main_layer]
        self.drawing.add(self.root_group)
        self.root_group.add(self.debug_layer)
        self.root_group.add(self.main_layer)

    def add(
        self,
        element: Union[svgwrite.base.BaseElement, svgwrite.elementfactory.ElementBuilder],
        add_to_debug_layer: bool = False,
    ) -> Union[svgwrite.base.BaseElement, svgwrite.elementfactory.ElementBuilder]:
        """Add a SVG element as subelement either to main or debug layer.

        Args:
            element (svgwrite.base.BaseElement): append this SVG element
            debug (bool, optional): True if element should be added to debug layer. Defaults to False.

        Returns:
            svgwrite.base.BaseElement: the added element
        """
        if add_to_debug_layer:
            return self.debug_layer.add(element)
        return self.main_layer.add(element)

    def save_as(
        self,
        filename: str,
        include_debug_layer: bool = False,
        pretty: bool = False,
        indent: int = 2,
        compressed: bool = False,
    ):
        """Save as SVG file

        Args:
            filename (str): path and filename
            include_debug_layer (bool, optional): True if file should contain debug_layer. Defaults to False.
            pretty (bool, optional): True for easy readable output. Defaults to False.
            indent (int, optional): Indention if pretty is enabled. Defaults to 2 spaces.
            compressed (bool, optional): Save as compressed svgz-file. Defaults to False.
        """
        # Temporarily remove debug layer if not needed
        if not include_debug_layer:
            self.root_group.elements.remove(self.debug_layer)

        try:
            if compressed:
                # For compressed files, use text wrapper to avoid intermediate buffer
                with open(filename, "wb", buffering=65536) as svg_file:
                    with gzip.GzipFile(fileobj=svg_file, mode="wb", compresslevel=6) as gz_file:
                        with io.TextIOWrapper(gz_file, encoding="utf-8", write_through=False) as text_file:
                            self.drawing.write(text_file, pretty=pretty, indent=indent)
            else:
                # For uncompressed files, write directly with buffering
                with open(filename, "w", encoding="utf-8", buffering=65536) as svg_file:
                    self.drawing.write(svg_file, pretty=pretty, indent=indent)
        finally:
            # Always restore debug layer to maintain object state
            if not include_debug_layer:
                # Re-add debug layer at the beginning (before main_layer)
                self.root_group.elements.insert(0, self.debug_layer)

    @classmethod
    def create_page_a4(
        cls,
        viewbox_width_mm: float,
        viewbox_height_mm: float,
        viewbox_scale: float = 1.0,
    ) -> AvSvgPage:
        """
        Create a new page with A4 dimensions.

        The viewbox is centered horizontally and vertically on the page.
        The viewbox coordinates are scaled by viewbox_scale

        Args:
            viewbox_width_mm (float): The width of the viewbox in mm.
            viewbox_height_mm (float): The height of the viewbox in mm.
            viewbox_scale (float, optional): The scale factor for the viewbox.
                Defaults to 1.0. Usually 1.0 / viewbox_width_mm to scale x-coordinates between 0 and 1.

        Returns:
            AvSvgPage: A new page with A4 dimensions.
        """
        # DIN A4 page dimensions
        canvas_width_mm = 210  # DIN A4 page width in mm
        canvas_height_mm = 297  # DIN A4 page height in mm

        # calculate viewbox origin (top-left corner)
        viewbox_x_mm = (canvas_width_mm - viewbox_width_mm) / 2  # viewbox top-left corner
        viewbox_y_mm = (canvas_height_mm - viewbox_height_mm) / 2  # viewbox top-left corner

        svg_page = AvSvgPage(
            canvas_width_mm,
            canvas_height_mm,
            viewbox_x_mm,
            viewbox_y_mm,
            viewbox_width_mm,
            viewbox_height_mm,
            viewbox_scale,
        )

        return svg_page


def main():
    """Main"""

    output_filename = "data/output/example/svg/ave/example_PageSVG.svgz"

    # create a page with A4 dimensions and a viewbox in the middle of the page
    vb_width_mm = 170  # 105  # viewbox width in mm
    vb_height_mm = 120  # 148.5  # viewbox height in mm
    vb_scale = 1.0 / vb_width_mm  # scale viewbox so that x-coordinates are between 0 and 1
    svg_page = AvSvgPage.create_page_a4(vb_width_mm, vb_height_mm, vb_scale)
    stroke_width_mm = 0.1  # stroke width = 0.1 mm

    # define a path that describes the outline of the viewbox
    svg_page.add(
        svg_page.drawing.path(
            d=(
                f"M 0 0 "
                f"L {vb_scale * vb_width_mm} 0 "  # = (1.0, 0.0)
                f"L {vb_scale * vb_width_mm} {vb_scale * vb_height_mm} "
                f"L 0 {vb_scale * vb_height_mm} "
                f"Z"
            ),
            stroke="black",
            stroke_width=stroke_width_mm * vb_scale,
            fill="none",
        )
    )

    # define a red path that describes a box in the left-bottom corner of the viewbox
    svg_page.add(
        svg_page.drawing.path(
            d=(
                f"M 0 0 "
                f"L {vb_scale * vb_width_mm * 0.1} 0 "
                f"L {vb_scale * vb_width_mm * 0.1} {vb_scale * vb_height_mm * 0.1} "
                f"L 0 {vb_scale * vb_height_mm * 0.1} "
                f"Z"
            ),
            stroke="red",
            stroke_width=stroke_width_mm * vb_scale,
            fill="none",
        )
    )

    # define a green path that describes a box in the right-top corner of the viewbox
    svg_page.add(
        svg_page.drawing.path(
            d=(
                f"M {vb_scale * vb_width_mm * 0.9} {vb_scale * vb_height_mm * 0.9} "
                f"L {vb_scale * vb_width_mm} {vb_scale * vb_height_mm * 0.9} "
                f"L {vb_scale * vb_width_mm} {vb_scale * vb_height_mm} "
                f"L {vb_scale * vb_width_mm * 0.9} {vb_scale * vb_height_mm} "
                f"Z"
            ),
            stroke="green",
            stroke_width=stroke_width_mm * vb_scale,
            fill="none",
        )
    )

    # define a blue path that describes a box in the middle of the viewbox
    half_width_mm = 0.5
    svg_page.add(
        svg_page.drawing.path(
            d=(
                f"M {vb_scale * (vb_width_mm * 0.5 - half_width_mm)} {vb_scale * (vb_height_mm * 0.5 -half_width_mm)} "
                f"L {vb_scale * (vb_width_mm * 0.5 - half_width_mm)} {vb_scale * (vb_height_mm * 0.5 +half_width_mm)} "
                f"L {vb_scale * (vb_width_mm * 0.5 + half_width_mm)} {vb_scale * (vb_height_mm * 0.5 +half_width_mm)} "
                f"L {vb_scale * (vb_width_mm * 0.5 + half_width_mm)} {vb_scale * (vb_height_mm * 0.5 -half_width_mm)} "
                f"Z"
            ),
            stroke="blue",
            stroke_width=stroke_width_mm * vb_scale,
            fill="none",
        ),
        True,
    )

    font_size = vb_scale * 3  # in mm

    # load a font and place letter L on lower left corner and letter T on upper right corner
    ttfont_filename = "fonts/RobotoFlex-VariableFont_GRAD,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf"
    ttfont = FontHelper.instantiate_ttfont(TTFont(ttfont_filename), {"wght": 400})
    glyph_factory = AvGlyphCachedSourceFactory(AvGlyphFromTTFontFactory(ttfont))
    avfont = AvFont(glyph_factory)

    glyph = avfont.get_glyph("L")
    letter = AvSingleGlyphLetter(glyph, font_size / avfont.props.units_per_em, 0.0, 0.0, Align.LEFT)
    svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
    svg_page.add(svg_path)

    glyph = avfont.get_glyph("T")
    letter = AvSingleGlyphLetter(glyph, font_size / avfont.props.units_per_em, align=Align.RIGHT)
    letter.xpos = 1.0 - letter.advance_width
    letter.ypos = vb_scale * vb_height_mm - letter.height

    svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
    svg_page.add(svg_path)

    # Save the SVG file
    print(f"save file {output_filename} ...")
    svg_page.save_as(output_filename, include_debug_layer=True, pretty=True, indent=2, compressed=True)
    print("save done.")


if __name__ == "__main__":
    main()
