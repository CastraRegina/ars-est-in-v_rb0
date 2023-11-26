"""Creates an image with 10 different "gray" squares from white to black.
"""
from PIL import Image
import numpy as np


OUTPUT_FILENAME = "data/output/example/png/test_board_10grays.png"

WIDTH_OF_SQUARE = 100  # width [pixel] of one square
# BOARD_LAYOUT: 0=black, 9=white
BOARD_LAYOUT = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 9, 9, 9, 9, 9, 0],
    [0, 9, 1, 5, 2, 9, 0],
    [0, 9, 8, 0, 6, 9, 0],
    [0, 9, 4, 7, 3, 9, 0],
    [0, 9, 9, 9, 9, 9, 0],
    [0, 0, 0, 0, 0, 0, 0],
]
COLOR_SCALE = 255 / 9  # scale factor to scale max color value of board to white

# Retrieve height and width of the layout
BOARD_HEIGHT = len(BOARD_LAYOUT)
BOARD_WIDTH = len(BOARD_LAYOUT[0])


def main():
    """Creates an image with 10 different "gray" squares from white to black."""
    # Create a blank image with a white background
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3))

    # Set the squares of the board to certain gray values
    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            gray = BOARD_LAYOUT[i][j] * COLOR_SCALE
            board[i, j] = [gray, gray, gray]

    # Create a PIL image from the numpy array
    image = Image.fromarray(np.uint8(board))

    # Resize the image to X*X pixels for each square
    image = image.resize(
        (BOARD_WIDTH * WIDTH_OF_SQUARE, BOARD_HEIGHT * WIDTH_OF_SQUARE),
        Image.Resampling.NEAREST,
    )

    # Save the image as a png file
    image.save(OUTPUT_FILENAME)
    print(f"{OUTPUT_FILENAME} saved.")


if __name__ == "__main__":
    main()
