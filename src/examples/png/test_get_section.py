# from PIL import Image
import PIL.Image
# import Image from PIL
import numpy as np


class AVImage:
    def __init__(self, filename: str, convert: str = "L"):
        self.image = None
        self.width = 0
        self.height = 0
        with PIL.Image.open(filename) as self.image:
            if convert:
                self.image = self.image.convert(convert)
            (self.width, self.height) = self.image.size

    def section(x: float, y: float,
                width: float, height: float) -> PIL.Image:
        # relative to width 0..1
        if not self.image:
            return None
        # TODO: x,y,width,height min(0) max(1)
        # TODO: round()

    def avg_value(x: float, y: float,
                  width: float, height: float) -> float:
        pass
        # TODO

    def max_value(x: float, y: float,
                  width: float, height: float) -> float:
        pass
        # TODO

    def min_value(x: float, y: float,
                  width: float, height: float) -> float:
        pass
        # TODO


def extract_section(image_path, x, y, width, height):
    # Open the image file
    image = PIL.Image.open(image_path).convert('L')  # Convert to grayscale

    # Extract the section based on coordinates
    section = image.crop((x, y, x + width, y + height))

    return section


def calculate_average_gray_value(image_path, x, y, width, height):
    # Extract the section from the image
    section = extract_section(image_path, x, y, width, height)

    # Convert the section to a numpy array
    section_array = np.array(section)

    # Calculate the average gray value
    average_gray_value = np.mean(section_array)

    return average_gray_value


def save_section_as_image(section, output_path):
    # Save the section as a PNG file
    section.save(output_path)


# Example usage
image_path = 'data/output/example/png/test_board_10grays.png'
x = 0  # X-coordinate of the top-left corner of the section
y = 0   # Y-coordinate of the top-left corner of the section
width = 700   # Width of the section
height = 700  # Height of the section
output_path = 'section_output.png'

# Extract the section from the image
section = extract_section(image_path, x, y, width, height)

# Save the section as a PNG file
save_section_as_image(section, output_path)

# Calculate the average gray value of the section
average_gray_value = calculate_average_gray_value(
    image_path, x, y, width, height)
print(f"The average gray value of the section is: {average_gray_value}")
print(f"The section has been saved as: {output_path}")

image = AVImage(image_path)
