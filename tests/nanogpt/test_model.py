"""
Unit tests for the GPT model
"""

import unittest

import torch

from nanogpt.model import GPT, GPTConfig


class TestGPTModel(unittest.TestCase):
    """
    Test cases for the GPT model
    """

    def setUp(self):
        # Create a small model for testing
        self.config = GPTConfig(
            block_size=128, vocab_size=100, n_layer=2, n_head=4, n_embd=128, dropout=0.0, bias=True
        )
        self.model = GPT(self.config)

    def test_model_init(self):
        """Test model initialization"""
        self.assertEqual(self.model.config.block_size, 128)
        self.assertEqual(self.model.config.vocab_size, 100)
        self.assertEqual(self.model.config.n_layer, 2)
        self.assertEqual(self.model.config.n_head, 4)
        self.assertEqual(self.model.config.n_embd, 128)

    def test_forward(self):
        """Test forward pass"""
        batch_size = 2
        seq_len = 10
        device = "cpu"

        # Create random input
        idx = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=device)

        # Forward pass
        logits, loss = self.model(idx, targets)

        # Check output shapes
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertEqual(loss.dim(), 0)  # Loss should be a scalar

    def test_generate(self):
        """Test text generation"""
        batch_size = 1
        seq_len = 5
        max_new_tokens = 10
        device = "cpu"

        # Create random input
        idx = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=device)

        # Generate text
        self.model.eval()
        output = self.model.generate(idx, max_new_tokens=max_new_tokens)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len + max_new_tokens))

    def test_crop_block_size(self):
        """Test block size cropping"""
        new_block_size = 64
        self.model.crop_block_size(new_block_size)

        # Check if block size was updated
        self.assertEqual(self.model.config.block_size, new_block_size)
        self.assertEqual(self.model.transformer.wpe.weight.shape[0], new_block_size)

    def test_get_num_params(self):
        """Test parameter counting"""
        num_params = self.model.get_num_params()
        num_params_no_embed = self.model.get_num_params(non_embedding=True)

        # Check that we have parameters
        self.assertGreater(num_params, 0)
        # Check that non_embedding count is less than total count
        self.assertLess(num_params_no_embed, num_params)


if __name__ == "__main__":
    unittest.main()
