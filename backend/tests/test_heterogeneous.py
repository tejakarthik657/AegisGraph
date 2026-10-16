"""
AegisGraph — Heterogeneous GNN Tests.

Tests the heterogeneous model module.
Skips tests when PyTorch is unavailable or broken.
"""

import unittest

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

SKIP_MSG = "PyTorch not available or DLL load failed"


class TestHeterogeneousModel(unittest.TestCase):
    """Test heterogeneous GNN model architecture."""

    @unittest.skipUnless(TORCH_AVAILABLE, SKIP_MSG)
    def test_model_module_imports(self):
        """Verify the heterogeneous model module can be imported."""
        from app.ml_extensions.heterogeneous import model
        self.assertTrue(hasattr(model, '__file__'))

    @unittest.skipUnless(TORCH_AVAILABLE, SKIP_MSG)
    def test_preprocessing_module_imports(self):
        """Verify the preprocessing module can be imported."""
        from app.ml_extensions.heterogeneous import preprocessing
        self.assertTrue(hasattr(preprocessing, '__file__'))

    @unittest.skipUnless(TORCH_AVAILABLE, SKIP_MSG)
    def test_train_module_imports(self):
        """Verify the train module can be imported."""
        from app.ml_extensions.heterogeneous import train
        self.assertTrue(hasattr(train, '__file__'))


class TestHybridModel(unittest.TestCase):
    """Test hybrid model components."""

    @unittest.skipUnless(TORCH_AVAILABLE, SKIP_MSG)
    def test_hybrid_model_import(self):
        from app.ml_extensions.hybrid.model import HybridFraudDetector
        self.assertTrue(callable(HybridFraudDetector))

    @unittest.skipUnless(TORCH_AVAILABLE, SKIP_MSG)
    def test_hybrid_model_creation(self):
        from app.ml_extensions.hybrid.model import HybridFraudDetector

        model = HybridFraudDetector(
            tgn_embedding_dim=32,
            hgnn_embedding_dim=32,
            hidden_dim=64,
        )

        tgn_embed = torch.randn(4, 32)
        hgnn_embed = torch.randn(4, 32)

        output = model(tgn_embed, hgnn_embed)
        self.assertEqual(output.shape, (4, 2))

    @unittest.skipUnless(TORCH_AVAILABLE, SKIP_MSG)
    def test_hybrid_predict_proba(self):
        from app.ml_extensions.hybrid.model import HybridFraudDetector

        model = HybridFraudDetector(
            tgn_embedding_dim=16,
            hgnn_embedding_dim=16,
        )

        tgn_embed = torch.randn(8, 16)
        hgnn_embed = torch.randn(8, 16)

        probs = model.predict_proba(tgn_embed, hgnn_embed)
        self.assertEqual(probs.shape, (8,))
        self.assertTrue(torch.all(probs >= 0.0))
        self.assertTrue(torch.all(probs <= 1.0))


if __name__ == "__main__":
    unittest.main()
