"""Creates an image with 10 different "gray" squares from white to black."""

import numpy as np
from PIL import Image

WIDTH_OF_SQUARE = 100  # width [pixel] of one square

OUTPUT_FILENAMES = [
    "data/output/example/png/test_board_07x07_10grays.png",
    "data/output/example/png/test_board_09x09_10grays.png",
]

# BOARD_LAYOUT: 0=black(0), 9=white(255)
BOARD_LAYOUTS = [
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 9, 9, 9, 9, 9, 0],
        [0, 9, 1, 5, 2, 9, 0],
        [0, 9, 8, 0, 6, 9, 0],
        [0, 9, 4, 7, 3, 9, 0],
        [0, 9, 9, 9, 9, 9, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [9, 9, 9, 9, 9, 9, 9, 9, 9],
        [9, 0, 0, 0, 0, 0, 0, 0, 9],
        [9, 0, 9, 9, 9, 9, 9, 0, 9],
        [9, 0, 9, 1, 5, 2, 9, 0, 9],
        [9, 0, 9, 8, 0, 6, 9, 0, 9],
        [9, 0, 9, 4, 7, 3, 9, 0, 9],
        [9, 0, 9, 9, 9, 9, 9, 0, 9],
        [9, 0, 0, 0, 0, 0, 0, 0, 9],
        [9, 9, 9, 9, 9, 9, 9, 9, 9],
    ],
]
COLOR_SCALE = 255 / 9  # scale factor to scale max color value of board to white


def main():
    """Creates images with 10 different 'gray' squares from white to black."""
    # Iterate over all layouts and create corresponding images
    for board_layout, output_filename in zip(BOARD_LAYOUTS, OUTPUT_FILENAMES):
        # Retrieve height and width of the current layout
        board_height = len(board_layout)
        board_width = len(board_layout[0])

        # Create a blank image with a white background
        board = np.zeros((board_height, board_width, 3))

        # Set the squares of the board to certain gray values
        for i in range(board_height):
            for j in range(board_width):
                gray = board_layout[i][j] * COLOR_SCALE
                board[i, j] = [gray, gray, gray]

        # Create a PIL image from the numpy array
        image = Image.fromarray(np.uint8(board))

        # Resize the image to X*X pixels for each square
        image = image.resize(
            (board_width * WIDTH_OF_SQUARE, board_height * WIDTH_OF_SQUARE),
            Image.Resampling.NEAREST,
        )

        # Save the image as a png file
        image.save(output_filename)
        print(f"{output_filename} saved.")


if __name__ == "__main__":
    main()
