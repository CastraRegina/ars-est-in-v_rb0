"""Creates a SVG file of a DIN A4 page
with a 140x100mm black outline of a rectangle in the middle of the page.
The strokewidth should be 0.1mm thick.
"""
import svgwrite

OUTPUT_FILE = "data/output/example/svg/din_a4_page_rectangle.svg"

CANVAS_WIDTH = 210  # DIN A4 page width in mm
CANVAS_HEIGHT = 297  # DIN A4 page height in mm
RECT_WIDTH = 140  # in mm
RECT_HEIGHT = 100  # in mm


def main():
    """Creates a SVG drawing object with the specified dimensions,
    then calculates the position of the rectangle on the page
    and adds it to the drawing.
    Finally, it saves the drawing to a SVG file.
    """
    # Set up the SVG canvas
    dwg = svgwrite.Drawing(OUTPUT_FILE, size=(CANVAS_WIDTH, CANVAS_HEIGHT))

    # Center the rectangle horizontally and vertically on the page
    rect_x = (CANVAS_WIDTH - RECT_WIDTH) / 2
    rect_y = (CANVAS_HEIGHT - RECT_HEIGHT) / 2

    # Draw the rectangle
    dwg.add(
        dwg.rect(
            insert=(rect_x, rect_y),
            size=(RECT_WIDTH, RECT_HEIGHT),
            stroke="black",
            stroke_width=0.1,
            fill="none",
        )
    )

    # Save the SVG file
    dwg.save()

    return 0


if __name__ == "__main__":
    main()
