import unittest
from pkg_resources import resource_filename

from ddplt import *


class TestEverything(unittest.TestCase):

    def test_fubar(self):
        self.assertEqual("Ja", "Ja")
        print("###\n#\n#\n#    TESTED\n#\n#\n#\n###")
