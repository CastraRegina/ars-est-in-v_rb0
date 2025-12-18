"""SVG page representation and layout utilities for document generation."""

from __future__ import annotations

import copy
import gzip
import io
from dataclasses import dataclass
from typing import Optional, Union

import svgwrite
import svgwrite.base
import svgwrite.container
import svgwrite.elementfactory
from fontTools.ttLib import TTFont
from svgwrite.extensions import Inkscape

from ave.common import Align
from ave.font import AvFont, AvFontProperties
from ave.fonttools import FontHelper
from ave.glyph import AvGlyphFromTTFontFactory, AvLetter


@dataclass
class AvSvgPage:
    """A page (canvas) described by SVG with a viewbox to draw inside.

    The viewbox has its own coordinate-system left-to-right and bottom-to-top.
    Contains groups/layers:
        - root       -- (group) just contains the y-flip and translation to bottom left
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
        drawing_for_save = self.assemble_tree(
            copy.deepcopy(self.drawing),
            copy.deepcopy(self.root_group),
            copy.deepcopy(self.main_layer),
            copy.deepcopy(self.debug_layer),
            include_debug_layer,
        )

        # setup IO:
        svg_buffer = io.StringIO()
        drawing_for_save.write(svg_buffer, pretty=pretty, indent=indent)
        output_data = svg_buffer.getvalue().encode("utf-8")
        if compressed:
            output_data = gzip.compress(output_data)

        # save file:
        with open(filename, "wb") as svg_file:
            svg_file.write(output_data)

    @classmethod
    def assemble_tree(
        cls,
        drawing: svgwrite.Drawing,
        root_group: svgwrite.container.Group,
        main_layer: svgwrite.container.Group,
        debug_layer: Optional[svgwrite.container.Group] = None,
        include_debug_layer: bool = False,
    ) -> svgwrite.Drawing:
        """Assemble a tree out of the given SVG elements.

        Args:
            drawing (svgwrite.Drawing): The main SVG drawing element.
            root_group (svgwrite.container.Group): The root group of the drawing.
            main_layer (svgwrite.container.Group): The main layer of the drawing.
            debug_layer (svgwrite.container.Group): The debug layer of the drawing.
            include_debug_layer (bool, optional): Include the debug layer in the tree. Defaults to False.

        Returns:
            svgwrite.Drawing: The given drawing with assembled SVG drawing elements.
        """
        drawing.add(root_group)
        if include_debug_layer and debug_layer:
            root_group.add(debug_layer)
        root_group.add(main_layer)
        return drawing

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
    glyph_factory = AvGlyphFromTTFontFactory(ttfont)
    avfont = AvFont(glyph_factory, AvFontProperties.from_ttfont(ttfont))

    glyph = avfont.get_glyph("L")
    letter = AvLetter.from_font_size_units_per_em(glyph, font_size, avfont.props.units_per_em, 0.0, 0.0, Align.LEFT)
    svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
    svg_page.add(svg_path)

    glyph = avfont.get_glyph("T")
    letter = AvLetter.from_font_size_units_per_em(glyph, font_size, avfont.props.units_per_em, align=Align.RIGHT)
    letter.xpos = 1.0 - letter.width
    letter.ypos = vb_scale * vb_height_mm - letter.height

    svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
    svg_page.add(svg_path)

    # Save the SVG file
    print(f"save file {output_filename} ...")
    svg_page.save_as(output_filename, include_debug_layer=True, pretty=True, indent=2, compressed=True)
    print("save done.")


if __name__ == "__main__":
    main()
