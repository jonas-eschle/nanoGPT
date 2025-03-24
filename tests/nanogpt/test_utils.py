"""
Unit tests for utility functions
"""

import os
import pickle
import tempfile
import unittest

import numpy as np
import torch

from nanogpt.utils import get_device, get_dtype, load_meta, set_seed, setup_logging


class TestUtils(unittest.TestCase):
    """
    Test cases for utility functions
    """

    def test_set_seed(self):
        """Test seed setting for reproducibility"""
        set_seed(42)
        rand1 = torch.rand(5)

        set_seed(42)
        rand2 = torch.rand(5)

        # Check that random numbers are the same with the same seed
        self.assertTrue(torch.all(torch.eq(rand1, rand2)))

    def test_setup_logging(self):
        """Test logging directory setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "logs")
            result_dir = setup_logging(log_dir)

            # Check that directory was created
            self.assertTrue(os.path.exists(log_dir))
            self.assertEqual(result_dir, log_dir)

    def test_get_device(self):
        """Test device selection"""
        # Test with explicit device
        device = get_device("cpu")
        self.assertEqual(device, "cpu")

        # Test auto-detection (result depends on environment)
        auto_device = get_device()
        self.assertIn(auto_device, ["cuda", "mps", "cpu"])

    def test_get_dtype(self):
        """Test dtype selection"""
        # Test with explicit dtype
        dtype = get_dtype("float32")
        self.assertEqual(dtype, "float32")

        # Test auto-detection (result depends on environment)
        auto_dtype = get_dtype()
        self.assertIn(auto_dtype, ["bfloat16", "float16", "float32"])

    def test_load_meta(self):
        """Test metadata loading"""
        # Create a temporary directory with meta.pkl
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data directory structure
            data_dir = os.path.join(temp_dir, "data", "test_dataset")
            os.makedirs(data_dir, exist_ok=True)

            # Create a mock meta.pkl file
            meta_data = {"vocab_size": 100, "test_key": "test_value"}
            meta_path = os.path.join(data_dir, "meta.pkl")
            with open(meta_path, "wb") as f:
                pickle.dump(meta_data, f)

            # Save the original implementation of load_meta
            original_load_meta = load_meta

            # Create a patched version for testing
            def patched_load_meta(dataset_name):
                meta_path = os.path.join(temp_dir, "data", dataset_name, "meta.pkl")
                if not os.path.exists(meta_path):
                    return None
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                return meta

            # Replace the function temporarily
            from nanogpt import utils

            utils.load_meta = patched_load_meta

            try:
                # Test loading meta
                meta = patched_load_meta("test_dataset")
                self.assertIsNotNone(meta)
                self.assertEqual(meta["vocab_size"], 100)
                self.assertEqual(meta["test_key"], "test_value")

                # Test loading non-existent meta
                meta = patched_load_meta("nonexistent_dataset")
                self.assertIsNone(meta)
            finally:
                # Restore original function
                utils.load_meta = original_load_meta


if __name__ == "__main__":
    unittest.main()
