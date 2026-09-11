"""Rollback ledger for CD Tuner's temporary parameter edits.

Track actual module objects rather than shared.sd_model, which may change during
refiner/model switches. Backups live on CPU and are released after restoration.
Diffusion weights can be restored at a stop step without undoing VAE saturation.
"""

from __future__ import annotations

from typing import Any


class TensorEditLedger:
    def __init__(self) -> None:
        self._entries: dict[tuple[int, str], tuple[Any, str, Any, Any, str]] = {}

    @property
    def pending(self) -> bool:
        return bool(self._entries)

    def capture(self, module: Any, name: str, *, group: str = "diffusion") -> None:
        import torch

        key = (id(module), name)
        if key in self._entries:
            return
        original = getattr(module, name)
        if not isinstance(original, torch.Tensor):
            raise ValueError(f"CD Tuner cannot edit non-tensor parameter {name}.")
        with torch.inference_mode():
            backup = original.detach().to(device="cpu", copy=True)
        self._entries[key] = (module, name, original, backup, group)

    def restore(self, *, group: str | None = None) -> None:
        import torch

        failures = []
        for key, (module, name, original, backup, entry_group) in reversed(list(self._entries.items())):
            if group is not None and entry_group != group:
                continue
            try:
                current = getattr(module, name, None)
                with torch.inference_mode():
                    if not isinstance(current, torch.Tensor) or current.shape != backup.shape:
                        setattr(module, name, original)
                        current = original
                    current.copy_(backup.to(device=current.device, dtype=current.dtype))
                del self._entries[key]
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError(
                "CD Tunerの重みを完全に復元できません。生成を中止してモデルを再読み込みしてください。"
            ) from failures[0]
