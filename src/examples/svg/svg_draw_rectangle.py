"""Creates a SVG file of a DIN A4 page (portrait format)
with a 140x100mm outline of a rectangle in the middle of the page.
The stroke width should be 0.1mm thick.
"""
import svgwrite

OUTPUT_FILE = "data/output/example/svg/din_a4_page_rectangle.svg"

CANVAS_UNIT = "mm"  # Units for CANVAS dimensions
CANVAS_WIDTH = 210  # DIN A4 page width in mm
CANVAS_HEIGHT = 297  # DIN A4 page height in mm

RECT_WIDTH = 140  # rectangle width in mm
RECT_HEIGHT = 100  # rectangle height in mm

VB_RATIO = 1 / RECT_WIDTH  # multiply each dimension with this ratio


def main():
    """Creates a SVG drawing object with the specified dimensions,
    then calculates the position of the rectangle on the page
    and adds it to the drawing.
    Finally, it saves the drawing to a SVG file.
    """

    # Center the rectangle horizontally and vertically on the page
    vb_w = VB_RATIO * CANVAS_WIDTH
    vb_h = VB_RATIO * CANVAS_HEIGHT
    vb_x = -VB_RATIO * (CANVAS_WIDTH - RECT_WIDTH) / 2
    vb_y = -VB_RATIO * (CANVAS_HEIGHT - RECT_HEIGHT) / 2

    # Set up the SVG canvas:
    # Define viewBox so that "1" is the width of the rectangle
    # Multiply a dimension with "VB_RATIO" to get the size regarding viewBox
    dwg = svgwrite.Drawing(OUTPUT_FILE,
                           size=(f"{CANVAS_WIDTH}mm", f"{CANVAS_HEIGHT}mm"),
                           viewBox=(f"{vb_x} {vb_y} {vb_w} {vb_h}")
                           )

    # Draw the rectangle
    dwg.add(
        dwg.rect(
            insert=(0, 0),
            size=(VB_RATIO*RECT_WIDTH, VB_RATIO*RECT_HEIGHT),  # = (1.0, xxxx)
            stroke="black",
            stroke_width=0.1*VB_RATIO,
            fill="none"
        )
    )

    # Save the SVG file
    dwg.saveas(OUTPUT_FILE, pretty=True, indent=2)


if __name__ == "__main__":
    main()
