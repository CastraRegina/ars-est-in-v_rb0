"""Representation of a page described by SVG"""

from __future__ import annotations

import copy
import gzip
import io
from dataclasses import dataclass
from typing import Dict, Optional, Union

import svgwrite
import svgwrite.base
import svgwrite.container
import svgwrite.elementfactory
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from svgwrite.extensions import Inkscape

from av.glyph import AVFont, AVGlyph


@dataclass
class AvPageSVG:
    """A page (canvas) described by SVG with a viewbox to draw inside

    Contains several layers:
        - main   -- editable->locked=False  --  hidden->display="block"
        - debug  -- editable->locked=False  --  hidden->display="none"
            - glyph
                - bounding_box
                - em_width
                - font_ascent_descent
                - sidebearing
            - background
    """

    drawing: svgwrite.Drawing
    _inkscape: Inkscape  # extension to support layers

    layer_main: svgwrite.container.Group
    layer_debug: svgwrite.container.Group
    layer_debug_glyph: svgwrite.container.Group
    layer_debug_glyph_bounding_box: svgwrite.container.Group  # red
    layer_debug_glyph_em_width: svgwrite.container.Group  # blue
    layer_debug_glyph_font_ascent_descent: svgwrite.container.Group  # green
    layer_debug_glyph_sidebearing: svgwrite.container.Group  # yellow, orange
    layer_debug_background: svgwrite.container.Group

    class Helper:
        """Helper-class mainly to provide staticmethods for handling Glyphs"""

        @staticmethod
        def add_glyph_sidebearing(page: AvPageSVG, glyph: AVGlyph, x_pos: float, y_pos: float, font_size: float):
            """Add a box to debug layer showing glyph's sidebearing left and right"""
            sb_left = glyph.real_sidebearing_left(font_size)
            sb_right = glyph.real_sidebearing_right(font_size)

            rect_bb = glyph.rect_bounding_box(x_pos, y_pos, font_size)
            rect = (x_pos, rect_bb[1], sb_left, rect_bb[3])
            page.layer_debug_glyph_sidebearing.add(AVGlyph.svg_rect(page.drawing, rect, "none", 0, fill="yellow"))

            rect = (
                x_pos + glyph.real_width(font_size) - sb_right,
                rect_bb[1],
                sb_right,
                rect_bb[3],
            )
            page.layer_debug_glyph_sidebearing.add(AVGlyph.svg_rect(page.drawing, rect, "none", 0, fill="orange"))

        @staticmethod
        def add_glyph_font_ascent_descent(
            page: AvPageSVG, glyph: AVGlyph, x_pos: float, y_pos: float, font_size: float
        ):
            """Add a box to debug layer showing font's ascent and descent"""
            stroke_width = glyph.real_dash_thickness(font_size)
            rect = glyph.rect_font_ascent_descent(x_pos, y_pos, font_size)
            page.layer_debug_glyph_font_ascent_descent.add(
                AVGlyph.svg_rect(page.drawing, rect, "green", 0.3 * stroke_width)
            )

        @staticmethod
        def add_glyph_em_width(
            page: AvPageSVG,
            glyph: AVGlyph,
            x_pos: float,
            y_pos: float,
            font_size: float,
            ascent: float,
            descent: float,
        ):
            """Add a box to debug layer showing GlyphSet's width"""
            stroke_width = glyph.real_dash_thickness(font_size)
            rect = glyph.rect_em_width(x_pos, y_pos, ascent, descent, font_size)
            page.layer_debug_glyph_em_width.add(AVGlyph.svg_rect(page.drawing, rect, "blue", 0.2 * stroke_width))

        @staticmethod
        def add_glyph_bounding_box(page: AvPageSVG, glyph: AVGlyph, x_pos: float, y_pos: float, font_size: float):
            """Add glyph's bounding box to debug layer"""
            stroke_width = glyph.real_dash_thickness(font_size)
            rect = glyph.rect_bounding_box(x_pos, y_pos, font_size)
            page.layer_debug_glyph_bounding_box.add(AVGlyph.svg_rect(page.drawing, rect, "red", 0.1 * stroke_width))

    def __init__(
        self,
        canvas_width_mm: float,
        canvas_height_mm: float,
        viewbox_x: float,
        viewbox_y: float,
        viewbox_width: float,
        viewbox_height: float,
    ):
        """Setup page defined by canvas width and height together with
        viewbox position, width and height.
        The viewbox is the area for drawing.

        Args:
            canvas_width_mm (float): _description_
            canvas_height_mm (float): _description_
            viewbox_x (float): _description_
            viewbox_y (float): _description_
            viewbox_width (float): _description_
            viewbox_height (float): _description_
        """

        # setup canvas & viewbox
        self.drawing: svgwrite.Drawing = svgwrite.Drawing(
            size=(f"{canvas_width_mm}mm", f"{canvas_height_mm}mm"),
            viewBox=(f"{viewbox_x} {viewbox_y} {viewbox_width} {viewbox_height}"),
            profile="full",
        )

        # use Inkscape extension to support layers
        self._inkscape: Inkscape = Inkscape(self.drawing)

        # define layers
        self.layer_debug = self._inkscape.layer(label="Layer debug", locked=False, display="none")
        self.layer_debug_background = self._inkscape.layer(label="Layer background", locked=True)
        self.layer_debug_glyph = self._inkscape.layer(label="Layer glyph", locked=True)
        self.layer_debug_glyph_sidebearing = self._inkscape.layer(label="Layer sidebearing", locked=True)
        self.layer_debug_glyph_font_ascent_descent = self._inkscape.layer(
            label="Layer font_ascent_descent", locked=True
        )
        self.layer_debug_glyph_em_width = self._inkscape.layer(label="Layer em_width", locked=True)
        self.layer_debug_glyph_bounding_box = self._inkscape.layer(label="Layer bounding_box", locked=True)
        self.layer_main = self._inkscape.layer(label="Layer main", locked=False)

        # build up layer hierarchy
        self.layer_debug.add(self.layer_debug_background)
        self.layer_debug.add(self.layer_debug_glyph)
        self.layer_debug_glyph.add(self.layer_debug_glyph_sidebearing)
        self.layer_debug_glyph.add(self.layer_debug_glyph_font_ascent_descent)
        self.layer_debug_glyph.add(self.layer_debug_glyph_em_width)
        self.layer_debug_glyph.add(self.layer_debug_glyph_bounding_box)

        # self.drawing.add(self.layer_debug)
        # self.drawing.add(self.layer_main)

    # def draw_path(
    #     self, path_string: str, **svg_properties
    # ) -> svgwrite.elementfactory.ElementBuilder:
    #     return self.drawing.path(path_string, **svg_properties)

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
            include_debug_layer (bool, optional): True if file should contain debug_layer.
                Defaults to False.
            pretty (bool, optional): True for easy readable output. Defaults to False.
            indent (int, optional): Indention if pretty is enabled. Defaults to 2 spaces.
            compressed (bool, optional): Save as compressed svgz-file. Defaults to False.
        """
        # copy drawing and include layer-copies:
        drawing = copy.deepcopy(self.drawing)
        if include_debug_layer:
            drawing.add(copy.deepcopy(self.layer_debug))
        drawing.add(copy.deepcopy(self.layer_main))

        # setup IO:
        svg_buffer = io.StringIO()
        drawing.write(svg_buffer, pretty=pretty, indent=indent)
        output_data = svg_buffer.getvalue().encode("utf-8")
        if compressed:
            output_data = gzip.compress(output_data)
        with open(filename, "wb") as svg_file:
            svg_file.write(output_data)

    def add_glyph(
        self,
        glyph: AVGlyph,
        x_pos: float,
        y_pos: float,
        font_size: float,
        ascent: Optional[float] = None,
        descent: Optional[float] = None,
    ):
        """Add a glyph as SVG element as subelement.
           Additionally associated boxes are added as debug-layer-subelements

        Args:
            glyph (AVGlyph): _description_
            x_pos (float): _description_
            y_pos (float): _description_
            font_size (float): _description_
            ascent (Optional[float], optional): _description_. Defaults to None.
            descent (Optional[float], optional): _description_. Defaults to None.
        """
        if not ascent:
            ascent = glyph.font_ascender()
        if not descent:
            descent = glyph.font_descender()
        AvPageSVG.Helper.add_glyph_em_width(self, glyph, x_pos, y_pos, font_size, ascent, descent)
        AvPageSVG.Helper.add_glyph_sidebearing(self, glyph, x_pos, y_pos, font_size)
        AvPageSVG.Helper.add_glyph_font_ascent_descent(self, glyph, x_pos, y_pos, font_size)
        AvPageSVG.Helper.add_glyph_bounding_box(self, glyph, x_pos, y_pos, font_size)
        self.add(glyph.svg_path(self.drawing, x_pos, y_pos, font_size))

    def add(
        self, element: Union[svgwrite.base.BaseElement, svgwrite.elementfactory.ElementBuilder]
    ) -> Union[svgwrite.base.BaseElement, svgwrite.elementfactory.ElementBuilder]:
        """Add an SVG element as subelement.

        Args:
            element (svgwrite.base.BaseElement): append this SVG element

        Returns:
            svgwrite.base.BaseElement: the added element
        """
        return self.layer_main.add(element)


def main():
    """Main"""

    output_filename = "data/output/example/svg/example_PageSVG.svg"

    canvas_width = 210  # DIN A4 page width in mm
    canvas_height = 297  # DIN A4 page height in mm

    rect_vb_width = 150  # rectangle viewbox width in mm
    rect_vb_height = 150  # rectangle viewbox height in mm

    vb_ratio = 1 / rect_vb_width  # multiply each dimension with this ratio

    # Center the rectangle horizontally and vertically on the page
    vb_w = vb_ratio * canvas_width
    vb_h = vb_ratio * canvas_height
    vb_x = -vb_ratio * (canvas_width - rect_vb_width) / 2
    vb_y = -vb_ratio * (canvas_height - rect_vb_height) / 2

    # Set up the SVG canvas:
    #   Define viewBox so that "1" is the width of the rectangle
    #   Multiply a dimension with "vb_ratio" to get the size regarding viewBox
    svg_output = AvPageSVG(canvas_width, canvas_height, vb_x, vb_y, vb_w, vb_h)

    # Draw the rectangle
    svg_output.add(
        svg_output.drawing.rect(
            insert=(0, 0),
            size=(vb_ratio * rect_vb_width, vb_ratio * rect_vb_height),  # = (1.0, xxxx)
            stroke="black",
            stroke_width=0.1 * vb_ratio,
            fill="none",
        )
    )

    # Add some text
    def instantiate_font(ttfont: TTFont, values: Dict[str, float]) -> AVFont:
        # values {"wght": 700, "wdth": 25, "GRAD": 100}
        axes_values = AVFont.default_axes_values(ttfont)
        axes_values.update(values)
        ttfont = instancer.instantiateVariableFont(ttfont, axes_values)
        return AVFont(ttfont)

    font_filename = "fonts/RobotoFlex-VariableFont_GRAD,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf"
    font_size = vb_ratio * 3  # in mm
    text = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        + "abcdefghijklmnopqrstuvwxyz "
        + "ÄÖÜ äöü ß€µ@²³~^°\\ 1234567890 "
        + ',.;:+-*#_<> !"§$%&/()=?{}[]'
    )

    ttfont = TTFont(font_filename)
    avfont = instantiate_font(ttfont, {"wght": 800})
    x_pos = 0
    y_pos = 0.1
    for character in text:
        glyph = avfont.glyph(character)
        svg_output.add_glyph(glyph, x_pos, y_pos, font_size)
        x_pos += glyph.real_width(font_size)

    # Save the SVG file
    print("save...")
    svg_output.save_as(output_filename + "z", include_debug_layer=True, pretty=True, indent=2, compressed=True)
    print("save done.")

    print(rect_vb_height * vb_ratio)


if __name__ == "__main__":
    main()
