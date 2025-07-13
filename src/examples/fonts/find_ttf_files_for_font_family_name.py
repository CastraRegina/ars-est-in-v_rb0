"""
This module provides a function for finding TTF files of installed fonts on a
Linux system.

The function `find_ttf_files` takes a font family name as input and returns a
list of TTF file paths of installed fonts that match the given family name. If
no TTF file is found, the function returns an empty list.
"""

import subprocess
from typing import List


def find_ttf_files(font_family_name: str) -> List[str]:
    """
    Returns a list of TTF file paths of installed fonts with the given family
    name. If no TTF file is found, returns an empty list.

    Args:
    - font_family_name (str): The name of the font family to search for.

    Returns:
    - List[str]: A list of TTF file paths of installed fonts that match the
    given family name.
    """

    # Build the fc-list command to search for the given font family name and
    # output the font file path
    cmd = ["fc-list", "--format=%{file}\\n", f":family={font_family_name}"]

    # Run the fc-list command and capture the output
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    output = result.stdout.decode().strip()

    # If output is not empty, add all TTF files to the list and return it
    ttf_files: List[str] = []
    if output:
        for font_file in output.split("\n"):
            if font_file.endswith(".ttf"):
                ttf_files.append(font_file)

    # Return the list of TTF files
    return ttf_files


def main():
    """Main function to demonstrate the usage of find_ttf_files."""
    font_family_name = "Free Sans"  # Example font family name
    ttf_files = find_ttf_files(font_family_name)

    if ttf_files:
        print(f'TTF file(s) for font "{font_family_name}":')
        for ttf_file in ttf_files:
            print("   ", ttf_file)
    else:
        print(f"No TTF files found for font family '{font_family_name}'.")


if __name__ == "__main__":
    main()
