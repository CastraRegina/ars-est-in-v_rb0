"""Unittests"""

import unittest

import av.page

# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


class TestPageSvg(unittest.TestCase):
    """Test class"""

    def test_main(self):
        """A test function"""
        # do something useful here ...
        av.page.main()


if __name__ == "__main__":
    unittest.main()
