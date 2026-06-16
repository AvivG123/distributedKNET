"""PyTorch Lightning callbacks for training."""

import torch
import pytorch_lightning as pl


class NanInfChecker(pl.Callback):
    """Callback to detect NaN/Inf values in gradients."""

    def on_after_backward(self, trainer, pl_module):
        """Check for NaN/Inf in gradients after backward pass."""
        for name, param in pl_module.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(f"NaN/Inf detected in gradient of {name}")
        if hasattr(trainer, 'batch'):
            feats, _, _, _ = trainer.batch
            if not torch.isfinite(feats).all():
                raise RuntimeError("Non-finite values detected in input batch")


class SkipNaNBatch(pl.Callback):
    """Callback to skip batches with NaN gradients."""

    def on_after_backward(self, trainer, pl_module):
        """Skip batch if NaN detected in any gradient."""
        if any(torch.isnan(param.grad).any() for param in pl_module.parameters() if param.grad is not None):
            trainer.should_stop = False
            trainer.accumulated_batches = 0
            pl_module.zero_grad()

