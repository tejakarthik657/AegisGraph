"""
AegisGraph — Hybrid Model Tests.
"""

import unittest

import numpy as np


class TestHybridPreprocessing(unittest.TestCase):
    """Test hybrid preprocessing alignment."""

    def test_align_embeddings_import(self):
        from app.ml_extensions.hybrid.preprocessing import align_embeddings
        self.assertTrue(callable(align_embeddings))


class TestHybridTraining(unittest.TestCase):
    """Test hybrid training pipeline."""

    def test_train_hybrid_import(self):
        from app.ml_extensions.hybrid.train import train_hybrid
        self.assertTrue(callable(train_hybrid))


if __name__ == "__main__":
    unittest.main()
