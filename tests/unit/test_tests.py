import unittest

import pyiron_workflow_atomistics


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = pyiron_workflow_atomistics.__version__
        print(version)
        self.assertTrue(version.startswith("0"))
