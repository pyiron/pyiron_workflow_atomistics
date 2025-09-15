"""
Unit tests for pyiron_workflow_atomistics version handling.
"""

import unittest

import pyiron_workflow_atomistics
import pyiron_workflow_atomistics._version as version_module


class TestVersion(unittest.TestCase):
    """Test version handling functionality."""

    def test_version_attribute(self):
        """Test that __version__ attribute exists and is a string."""
        version = pyiron_workflow_atomistics.__version__
        self.assertIsInstance(version, str)
        self.assertGreater(len(version), 0)

    def test_version_format(self):
        """Test that version follows expected format."""
        version = pyiron_workflow_atomistics.__version__

        # Version should start with a number or be "0+unknown"
        self.assertTrue(
            version.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
            or version == "0+unknown"
        )

    def test_get_versions_function(self):
        """Test get_versions function."""
        versions = version_module.get_versions()

        # Check that result is a dictionary
        self.assertIsInstance(versions, dict)

        # Check that required keys are present
        required_keys = ["version", "full-revisionid", "dirty", "error", "date"]
        for key in required_keys:
            self.assertIn(key, versions)

        # Check that version is a string
        self.assertIsInstance(versions["version"], str)

    def test_get_config_function(self):
        """Test get_config function."""
        config = version_module.get_config()

        # Check that result is a VersioneerConfig object
        self.assertIsInstance(config, version_module.VersioneerConfig)

        # Check that required attributes are set
        self.assertEqual(config.VCS, "git")
        self.assertEqual(config.style, "pep440-pre")
        self.assertEqual(config.tag_prefix, "pyiron_workflow_atomistics-")
        # self.assertEqual(config.parentdir_prefix, "pyiron_workflow_atomistics")
        self.assertEqual(
            config.versionfile_source, "pyiron_workflow_atomistics/_version.py"
        )

    def test_get_keywords_function(self):
        """Test get_keywords function."""
        keywords = version_module.get_keywords()

        # Check that result is a dictionary
        self.assertIsInstance(keywords, dict)

        # Check that required keys are present
        required_keys = ["refnames", "full", "date"]
        for key in required_keys:
            self.assertIn(key, keywords)

        # Check that values are strings
        for value in keywords.values():
            self.assertIsInstance(value, str)

    def test_register_vcs_handler_decorator(self):
        """Test VCS handler registration decorator."""

        # Test that decorator can be applied
        @version_module.register_vcs_handler("test_vcs", "test_method")
        def test_handler():
            return "test_result"

        # Check that handler was registered
        self.assertIn("test_vcs", version_module.HANDLERS)
        self.assertIn("test_method", version_module.HANDLERS["test_vcs"])
        self.assertEqual(
            version_module.HANDLERS["test_vcs"]["test_method"], test_handler
        )

    def test_NotThisMethod_exception(self):
        """Test NotThisMethod exception."""
        # Test that exception can be raised and caught
        with self.assertRaises(version_module.NotThisMethod):
            raise version_module.NotThisMethod("test message")

    def test_versions_from_parentdir_function(self):
        """Test versions_from_parentdir function."""
        # This function requires specific directory structure, so we test the interface
        # and error handling rather than the full functionality

        # Test with invalid parentdir_prefix
        with self.assertRaises(version_module.NotThisMethod):
            version_module.versions_from_parentdir(
                parentdir_prefix="nonexistent_prefix", root="/tmp", verbose=False
            )

    def test_render_functions(self):
        """Test render functions for different styles."""
        # Create test pieces with all commonly expected keys
        pieces = {
            "closest-tag": "1.0.0",
            "distance": 5,
            "short": "abc1234",
            "dirty": False,
            "long": "abcdef1234567890",
            "date": "2024-01-01T12:00:00",
            "error": None,
            "branch": "master",
            "node": "abcdef1234567890",  # might be needed
            "node-date": "2024-01-01T12:00:00+00:00",  # might be needed
        }

        # Test different render styles
        styles = [
            "pep440",
            "pep440-branch",
            "pep440-pre",
            "pep440-post",
            "pep440-post-branch",
            "pep440-old",
            "git-describe",
            "git-describe-long",
        ]

        for style in styles:
            result = version_module.render(pieces, style)
            # Add assertions here to verify the result
            self.assertIsNotNone(result)
            print(f"Style {style}: {result}")  # Optional: to see what gets rendered

    def test_render_invalid_style(self):
        """Test render function with invalid style."""
        pieces = {
            "closest-tag": "1.0.0",
            "distance": 5,
            "short": "abc1234",
            "dirty": False,
            "long": "abcdef1234567890",
            "date": "2024-01-01T12:00:00",
            "error": None,
            "branch": "master",
            "node": "abcdef1234567890",  # might be needed
            "node-date": "2024-01-01T12:00:00+00:00",  # might be needed
        }

        with self.assertRaises(ValueError):
            version_module.render(pieces, "invalid_style")

    def test_render_with_error(self):
        """Test render function with error in pieces."""
        pieces = {
            "error": "test error",
            "long": "abcdef1234567890",
            # Add other keys that might be needed
            "closest-tag": None,
            "distance": None,
            "short": None,
            "dirty": None,
            "date": None,
            "branch": None,
        }

        result = version_module.render(pieces, "pep440")

        # Should return error information
        self.assertEqual(result["version"], "unknown")
        self.assertEqual(result["error"], "test error")

    def test_plus_or_dot_function(self):
        """Test plus_or_dot function."""
        # Test with plus in closest-tag
        pieces_with_plus = {"closest-tag": "1.0.0+dev"}
        result = version_module.plus_or_dot(pieces_with_plus)
        self.assertEqual(result, ".")

        # Test without plus in closest-tag
        pieces_without_plus = {"closest-tag": "1.0.0"}
        result = version_module.plus_or_dot(pieces_without_plus)
        self.assertEqual(result, "+")

        # Test with no closest-tag
        pieces_no_tag = {}
        result = version_module.plus_or_dot(pieces_no_tag)
        self.assertEqual(result, "+")

    def test_pep440_split_post_function(self):
        """Test pep440_split_post function."""
        # Test with post segment
        version, post = version_module.pep440_split_post("1.0.0.post1")
        self.assertEqual(version, "1.0.0")
        self.assertEqual(post, 1)

        # Test without post segment
        version, post = version_module.pep440_split_post("1.0.0")
        self.assertEqual(version, "1.0.0")
        self.assertIsNone(post)

        # Test with post0
        version, post = version_module.pep440_split_post("1.0.0.post0")
        self.assertEqual(version, "1.0.0")
        self.assertEqual(post, 0)


if __name__ == "__main__":
    unittest.main()
