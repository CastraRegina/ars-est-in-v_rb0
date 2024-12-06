"""Representation of a page described by SVG"""

from __future__ import annotations

import copy
import gzip
import io
from dataclasses import dataclass
from typing import Union

import svgwrite
import svgwrite.base
import svgwrite.container
import svgwrite.elementfactory
from svgwrite.extensions import Inkscape


@dataclass
class AvPageSvg:
    """A page (canvas) described by SVG with a viewbox to draw inside

    Contains layers:
        - main   -- editable->locked=False  --  hidden->display="block"
        - debug  -- editable->locked=False  --  hidden->display="none"
    """

    drawing: svgwrite.Drawing
    _inkscape: Inkscape  # extension to support layers

    layer_main: svgwrite.container.Group
    layer_debug: svgwrite.container.Group

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
        # (profile="full" to support numbers with more than 4 decimal digits)
        self.drawing: svgwrite.Drawing = svgwrite.Drawing(
            size=(f"{canvas_width_mm}mm", f"{canvas_height_mm}mm"),
            viewBox=(f"{viewbox_x} {viewbox_y} {viewbox_width} {viewbox_height}"),
            profile="full",
        )

        # use Inkscape extension to support layers
        self._inkscape: Inkscape = Inkscape(self.drawing)

        # define layers
        self.layer_main = self._inkscape.layer(label="main", locked=False)
        self.layer_debug = self._inkscape.layer(label="debug", locked=False, display="none")

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

        # save file:
        with open(filename, "wb") as svg_file:
            svg_file.write(output_data)

    def add(
        self,
        element: Union[svgwrite.base.BaseElement, svgwrite.elementfactory.ElementBuilder],
        debug: bool = False,
    ) -> Union[svgwrite.base.BaseElement, svgwrite.elementfactory.ElementBuilder]:
        """Add an SVG element as subelement.

        Args:
            element (svgwrite.base.BaseElement): append this SVG element
            debug (bool, optional): True if element should be added to debug layer. Defaults to False.

        Returns:
            svgwrite.base.BaseElement: the added element
        """
        if debug:
            return self.layer_debug.add(element)
        return self.layer_main.add(element)


def main():
    """Main"""

    output_filename = "data/output/example/svg/ave/example_PageSVG.svgz"

    canvas_width = 210  # DIN A4 page width in mm
    canvas_height = 297  # DIN A4 page height in mm

    viewbox_width = 150  # viewbox width in mm
    viewbox_height = 150  # viewbox height in mm

    viewbox_ratio = 1 / viewbox_width  # multiply each dimension with this ratio

    # Center the viewbox horizontally and vertically on the page
    vb_w = viewbox_ratio * canvas_width
    vb_h = viewbox_ratio * canvas_height
    vb_x = -viewbox_ratio * (canvas_width - viewbox_width) / 2
    vb_y = -viewbox_ratio * (canvas_height - viewbox_height) / 2

    # Set up the SVG canvas:
    #   Define viewBox so that "1" is the width of the viewbox
    #   Multiply a dimension with "vb_ratio" to get the size regarding viewBox
    svg_output = AvPageSvg(canvas_width, canvas_height, vb_x, vb_y, vb_w, vb_h)

    # Draw a rectangle to show the viewbox
    svg_output.add(
        svg_output.drawing.rect(
            insert=(0, 0),
            size=(viewbox_ratio * viewbox_width, viewbox_ratio * viewbox_height),  # = (1.0, xxxx)
            stroke="black",
            stroke_width=0.1 * viewbox_ratio,
            fill="none",
        )
    )

    # Save the SVG file
    print("save...")
    svg_output.save_as(output_filename, include_debug_layer=True, pretty=True, indent=2, compressed=True)
    print("save done.")


if __name__ == "__main__":
    main()
