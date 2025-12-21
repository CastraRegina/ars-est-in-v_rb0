"""Test module to run examples from the examples.ave package

The tests are run using pytest.
"""

import pytest  # pylint: disable=unused-import

import ave.page
from examples.ave import font_check_variable_fonts
from examples.fonts import (
    find_ttf_files_for_font_family_name,
    print_fonttools_glyph_metrics,
)

# def test_examples_page_svg():
#     """Test function for page_svg example"""
#     ave.page.main()
#     assert True


# def test_examples_font_check_variable_fonts():
#     """Test function for font_check_variable_fonts example"""
#     font_check_variable_fonts.main()
#     assert True


def test_examples_find_ttf_files_for_font_family_name():
    """Test function for find_ttf_files_for_font_family_name example"""

    find_ttf_files_for_font_family_name.main()
    assert True


def test_examples_print_fonttools_glyph_metrics():
    """Test function for print_fonttools_glyph_metrics example"""
    print_fonttools_glyph_metrics.main()
    assert True


# def test_main():
#     """A test function"""
#     assert True
