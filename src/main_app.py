"""Main application"""

from av.page import AvPageSvg


def main():
    """Main"""
    page = AvPageSvg(210, 297, 0, 0, 1, 1)

    page.save_as("main_app.svg")

    print("file saved.")


if __name__ == "__main__":
    main()
