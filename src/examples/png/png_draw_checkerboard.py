"""Creates checkerboard-like images.
width (= height) of a square and the
number of squares in one direction are specified.
"""
import os
from PIL import Image
import numpy as np


OUTPUT_FOLDER = "data/output/example/png/checkerboard"
OUTPUT_FILENAME = "checkerboard"
# WIDTH [pixel] of one square, NUMBER_OF_SQUARES in one direction:
WIDTH_AND_NUMBER_OF_SQUARES = {
    1: (2, 5, 10),
    2: (2, 5, 10),
    5: (2, 5, 10),
    10: (2, 5, 10),
    100: (2, 5, 10),
    1000: (2, 5),
    5000: (2, 4),
}


def main():
    """Creates checkerboard-like images."""
    # Iterate over the specified widths and number of squares:
    for width_of_square, tmp_numbers in WIDTH_AND_NUMBER_OF_SQUARES.items():
        for number_of_squares in tmp_numbers:
            # Create X*X array of 0 and 1
            board = np.array(
                [
                    [(i + j) % 2 for i in range(number_of_squares)]
                    for j in range(number_of_squares)
                ]
            )

            # Create an image from the array and set the color
            image = Image.fromarray(np.uint8(board) * 255)

            # Resize the image to X*X pixels for each square
            image = image.resize(
                (
                    number_of_squares * width_of_square,
                    number_of_squares * width_of_square,
                ),
                Image.Resampling.NEAREST,
            )

            # Create a suitable filename
            filename = (
                OUTPUT_FILENAME
                + f"__size_{(number_of_squares*width_of_square):06d}px"
                + f"__num_{number_of_squares:03d}"
                + f"__each_{width_of_square:04d}px.png"
            )

            # Save the image as a PNG file
            path_with_filename = os.path.join(OUTPUT_FOLDER, filename)
            image.save(path_with_filename)
            print(f"{path_with_filename} saved.")


if __name__ == "__main__":
    main()
