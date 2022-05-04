import unittest

import util


class UtilTest(unittest.TestCase):
    def test_extend_conv_spec(self):
        convolutions = [(512, 3), (512, 3, 2)]
        assert util.extend_conv_spec(convolutions) == (
            (512, 3, 1), (512, 3, 2))
