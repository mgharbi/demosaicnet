import os

from .modules import BayerDemosaick
from .modules import XTransDemosaick
from .mosaic import xtrans
from .mosaic import bayer

TEST_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_input.png")
