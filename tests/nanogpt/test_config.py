"""
Unit tests for the configuration handling
"""

import os
import tempfile
import unittest

from nanogpt.config import Config


class TestConfig(unittest.TestCase):
    """
    Test cases for the Config class
    """

    def test_init(self):
        """Test initialization with kwargs"""
        config = Config(a=1, b="test", c=True)
        self.assertEqual(config.a, 1)
        self.assertEqual(config.b, "test")
        self.assertEqual(config.c, True)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = Config(a=1, b="test", c=True)
        config_dict = config.to_dict()
        self.assertEqual(config_dict, {"a": 1, "b": "test", "c": True})

    def test_from_file(self):
        """Test loading from file"""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("a = 1\nb = 'test'\nc = True\n")
            config_file = f.name

        try:
            # Load config from file
            config = Config.from_file(config_file)

            # Check values
            self.assertEqual(config.a, 1)
            self.assertEqual(config.b, "test")
            self.assertEqual(config.c, True)
        finally:
            # Clean up
            os.unlink(config_file)

    def test_override_from_args(self):
        """Test overriding from command line args"""
        config = Config(a=1, b="test", c=True)

        # Override with args
        args = ["--a=2", "--b=override"]
        config.override_from_args(args)

        # Check values
        self.assertEqual(config.a, 2)
        self.assertEqual(config.b, "override")
        self.assertEqual(config.c, True)  # Unchanged

    def test_override_with_different_types(self):
        """Test overriding with different types"""
        config = Config(a=1, b="test", c=True)

        # Override with args of different types
        args = ["--a=2.5", "--c=False"]
        config.override_from_args(args)

        # Check values and types
        self.assertEqual(config.a, 2.5)
        self.assertEqual(config.b, "test")  # Unchanged
        self.assertEqual(config.c, False)


if __name__ == "__main__":
    unittest.main()
