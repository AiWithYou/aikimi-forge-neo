"""Job-scoped debug staging and publication."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .color import linear_rgb_to_image


class DebugWriter:
    def __init__(
        self,
        staging_directory: str | Path,
        *,
        stem: str,
        enabled: bool,
    ):
        self.root = Path(staging_directory) / "debug"
        self.stem = stem
        self.enabled = bool(enabled)
        self.files: list[Path] = []
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, suffix: str) -> Path:
        return self.root / f"{self.stem}{suffix}"

    def save_image(
        self, suffix: str, image: Image.Image | np.ndarray
    ) -> Path | None:
        if not self.enabled:
            return None
        if isinstance(image, np.ndarray):
            array = np.asarray(image)
            if array.ndim == 2:
                normalized = np.clip(array, 0.0, 1.0)
                image = Image.fromarray(
                    np.clip(np.rint(normalized * 255), 0, 255).astype(np.uint8),
                    mode="L",
                )
            else:
                image = linear_rgb_to_image(array)
        path = self._path(suffix)
        image.save(path, format="PNG")
        self.files.append(path)
        return path

    def save_json(self, suffix: str, value: Any) -> Path | None:
        if not self.enabled:
            return None
        path = self._path(suffix)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
            newline="\n",
        )
        self.files.append(path)
        return path

    def save_csv(
        self, suffix: str, rows: list[dict[str, Any]]
    ) -> Path | None:
        if not self.enabled or not rows:
            return None
        path = self._path(suffix)
        fieldnames = sorted(set().union(*(row.keys() for row in rows)))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.files.append(path)
        return path

    def publish(self, destination: str | Path) -> list[Path]:
        if not self.enabled:
            return []
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        published: list[Path] = []
        for source in self.files:
            destination_path = target / source.name
            shutil.copy2(source, destination_path)
            published.append(destination_path)
        return published
