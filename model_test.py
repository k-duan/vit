import unittest

import torch

from model import ImageTokenizer, Attention

class TestImageTokenizer(unittest.TestCase):
    def setUp(self):
        self._image_tokenizer = ImageTokenizer(patch_size=8, n_channels=3, embedding_dim=512)

    def testForward(self):
        test_input = torch.randn((4, 3, 32, 32), dtype=torch.float32)
        output = self._image_tokenizer(test_input)
        self.assertEqual(output.shape, (4, 16, 512))

class TestAttention(unittest.TestCase):
    def setUp(self):
        self._attention = Attention(embedding_dim=512, n_heads=8)

    def testForward(self):
        test_input = torch.randn((4, 16, 512), dtype=torch.float32)
        output = self._attention(test_input)
        self.assertEqual(output.shape, (4, 16, 512))

