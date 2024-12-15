"""Representation of a page described by SVG"""

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
from svgwrite.extensions import Inkscape


@dataclass
class AvSvgPage:
    """A page (canvas) described by SVG with a viewbox to draw inside

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
        viewbox_x: float,
        viewbox_y: float,
        viewbox_width: float,
        viewbox_height: float,
    ):
        """Initialize the SVG page with specified canvas and viewbox dimensions.
            The viewbox is the drawing area.

        Args:
            canvas_width_mm (float): The width of the canvas in millimeters.
            canvas_height_mm (float): The height of the canvas in millimeters.
            viewbox_x (float): The x-coordinate of the viewbox's starting point.
            viewbox_y (float): The y-coordinate of the viewbox's starting point.
            viewbox_width (float): The width of the viewbox.
            viewbox_height (float): The height of the viewbox.
        """
        print(f"Initializing page: {canvas_width_mm} {canvas_height_mm}")
        print(f"Initializing viewbox: {viewbox_x} {viewbox_y} {viewbox_width} {viewbox_height}")
        # Setup canvas and viewbox. profile="full" to support numbers with more than 4 decimal digits
        self.drawing: svgwrite.Drawing = svgwrite.Drawing(
            size=(f"{canvas_width_mm}mm", f"{canvas_height_mm}mm"),
            viewBox=(f"{viewbox_x} {viewbox_y} {viewbox_width} {viewbox_height}"),
            profile="full",
        )

        # Define root group with transformation to flip y-axis and set origin to bottom-left
        # self.root_group: svgwrite.container.Group = self.drawing.g(transform=f"scale(1,-1) translate(0,-1)")
        y_translate = -(viewbox_height / viewbox_width)
        # y_translate = -0.1
        print(f"y_translate: {y_translate}")
        # self.root_group: svgwrite.container.Group = self.drawing.g(transform=f"scale(1,-1) translate(0,{y_translate})")
        self.root_group: svgwrite.container.Group = self.drawing.g(transform=f"scale(1,1)")
        # transform=f"scale(1,-1) translate(0,{-canvas_height_mm})"
        # TODO: replace translate-1 with correct y-value

        # Initialize Inkscape extension for layer support
        self._inkscape: Inkscape = Inkscape(self.drawing)

        # Define layers
        self.main_layer = self._inkscape.layer(label="Main", locked=False)
        self.debug_layer = self._inkscape.layer(label="Debug", locked=False, display="none")

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
            svgwrite.Drawing: The assembled SVG drawing element.
        """
        drawing.add(root_group)
        if include_debug_layer and debug_layer:
            root_group.add(debug_layer)
        root_group.add(main_layer)
        return drawing

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
        drawing = self.assemble_tree(
            copy.deepcopy(self.drawing),
            copy.deepcopy(self.root_group),
            copy.deepcopy(self.main_layer),
            copy.deepcopy(self.debug_layer),
            include_debug_layer,
        )

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
        add_to_debug_layer: bool = False,
    ) -> Union[svgwrite.base.BaseElement, svgwrite.elementfactory.ElementBuilder]:
        """Add an SVG element as subelement.

        Args:
            element (svgwrite.base.BaseElement): append this SVG element
            debug (bool, optional): True if element should be added to debug layer. Defaults to False.

        Returns:
            svgwrite.base.BaseElement: the added element
        """
        if add_to_debug_layer:
            return self.debug_layer.add(element)
        return self.main_layer.add(element)


def main():
    """Main"""

    output_filename = "data/output/example/svg/ave/example_PageSVG.svgz"

    canvas_width = 210  # DIN A4 page width in mm
    canvas_height = 297  # DIN A4 page height in mm

    viewbox_width = 105  # viewbox width in mm
    viewbox_height = 148.5  # viewbox height in mm

    viewbox_ratio = 1 / viewbox_width  # multiply each dimension with this ratio

    # Center the viewbox horizontally and vertically on the page
    #   vb_x seen from viewbox left border
    vb_w = viewbox_ratio * canvas_width
    vb_h = viewbox_ratio * canvas_height
    vb_x = -viewbox_ratio * (canvas_width - viewbox_width) / 2
    vb_y = -viewbox_ratio * (canvas_height - viewbox_height) / 2
    # vb_y = 0  # viewbox_ratio * ((canvas_height - viewbox_height) / 2 + 2 * viewbox_height)
    # -viewbox_ratio * (canvas_height - viewbox_height) / 2
    # vb_y = (
    #     viewbox_ratio * viewbox_width
    #     - viewbox_ratio * (canvas_height - viewbox_height) / 2
    #     - viewbox_ratio * viewbox_height
    # )

    # Set up the SVG canvas:
    #   Define viewBox so that "1" is the width of the viewbox
    #   Multiply a dimension with "vb_ratio" to get the size regarding viewBox
    # vb_y = -viewbox_ratio * (canvas_height - viewbox_height) / 2

    svg_page = AvSvgPage(canvas_width, canvas_height, vb_x, vb_y, vb_w, vb_h)

    # define a path that describes the outline of the viewbox
    svg_page.add(
        svg_page.drawing.path(
            d=(
                f"M 0 0 "
                f"L {viewbox_ratio * viewbox_width} 0 "  # = (1.0, 0.0)
                f"L {viewbox_ratio * viewbox_width} {viewbox_ratio * viewbox_height} "
                f"L 0 {viewbox_ratio * viewbox_height} "
                f"Z"
            ),
            stroke="black",
            stroke_width=0.1 * viewbox_ratio,
            fill="none",
        )
    )

    # define a red path that describes a box in the left-bottom corner of the viewbox
    svg_page.add(
        svg_page.drawing.path(
            d=(
                f"M 0 0 "
                f"L {viewbox_ratio * viewbox_width * 0.1} 0 "
                f"L {viewbox_ratio * viewbox_width * 0.1} {viewbox_ratio * viewbox_height * 0.1} "
                f"L 0 {viewbox_ratio * viewbox_height * 0.1} "
                f"Z"
            ),
            stroke="red",
            stroke_width=0.1 * viewbox_ratio,
            fill="none",
        )
    )

    # define a green path that describes a box in the right-top corner of the viewbox
    svg_page.add(
        svg_page.drawing.path(
            d=(
                f"M {viewbox_ratio * viewbox_width * 0.9} {viewbox_ratio * viewbox_height * 0.9} "
                f"L {viewbox_ratio * viewbox_width} {viewbox_ratio * viewbox_height * 0.9} "
                f"L {viewbox_ratio * viewbox_width} {viewbox_ratio * viewbox_height} "
                f"L {viewbox_ratio * viewbox_width * 0.9} {viewbox_ratio * viewbox_height} "
                f"Z"
            ),
            stroke="green",
            stroke_width=0.1 * viewbox_ratio,
            fill="none",
        )
    )

    # Save the SVG file
    print("save...")
    svg_page.save_as(output_filename, include_debug_layer=True, pretty=True, indent=2, compressed=True)
    print("save done.")


if __name__ == "__main__":
    main()


# ########################################
# import svgwrite
# # Define the dimensions of the DIN A4 page in pixels
# page_width = 2480
# page_height = 3508

# # Define the dimensions of the viewbox
# viewbox_width = 1
# viewbox_height = page_height / page_width  # Maintain aspect ratio, approximately 1.414

# # Create an SVG drawing object with the specified page dimensions
# dwg = svgwrite.Drawing("triangle_viewbox.svg", profile="tiny", size=(page_width, page_height))

# # Set the viewbox to be centered on the DIN A4 page with half the width and height of the page
# dwg.viewbox(0, 0, viewbox_width, int(viewbox_height))

# # Define the points of the triangle
# points = [(0.5, viewbox_height), (0, viewbox_height / 2), (1, viewbox_height / 2)]

# # Create a group with scaling and translation transformation
# group = dwg.g(transform="scale(1, -1) translate(0, -1.414)")

# # Add the triangle to the group
# group.add(dwg.polygon(points, fill="none", stroke="black", stroke_width=0.01))

# # Draw a rectangle showing the size of the viewbox
# # Note: The rectangle should be positioned at (0, 0) with width=1 and height=viewbox_height

# rectangle = dwg.rect(
#     insert=(0, 0),
#     size=(viewbox_width, viewbox_height),
#     fill="none",
#     stroke="red",
#     stroke_width=0.01,
# )
# group.add(rectangle)

# # Add the group to the SVG drawing
# dwg.add(group)

# # Save the SVG file
# dwg.save()

# print("SVG file 'triangle_viewbox.svg' has been created.")
