"""PyTorch ``Dataset`` over the Mikha-Bench manifest + splits.

Design notes:

* Reads BGR from disk via OpenCV to match ``mikha.eval.corpus.EvalSample``
  exactly — no drift between train-time and eval-time preprocessing.
* Applies one Mikha-Aug degradation per training sample with probability
  ``aug_prob`` (default 0.5), sampled uniformly from the 5 canonical bench
  transforms. Val and test datasets pass ``aug_prob=0.0`` — evaluation
  augmentation lives in :mod:`mikha.eval.corpus` and stays out of this
  module's business.
* Augmentation determinism is per-sample: the RNG is seeded from
  ``(base_seed, row_id, epoch)`` so a re-run with the same seed produces
  byte-identical batches.
* Standard torchvision normalize follows ImageNet stats since the
  backbone is ImageNet-pretrained.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mikha.aug.pipeline import BENCH_TRANSFORMS
from mikha.bench import load_manifest, local_image_path
from mikha.eval.corpus import load_splits

# ImageNet stats — MobileNetV3 was pretrained with these.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_BENCH_AUG_NAMES = tuple(BENCH_TRANSFORMS.keys())


@dataclass(frozen=True)
class _Row:
    row_id: str
    image_path: Path
    label: int  # 1 = flood, 0 = nonflood


def _seed_for(row_id: str, epoch: int, base_seed: int) -> int:
    """Stable per-sample seed for degradation RNG (never re-derived elsewhere)."""
    digest = hashlib.sha256(f"{base_seed}|{row_id}|{epoch}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class FloodImageDataset:
    """PyTorch-compatible dataset — imports torch lazily.

    Yields ``(image_tensor, label)`` where ``image_tensor`` is a
    ``(3, H, W) float32`` tensor already normalized for the backbone and
    ``label`` is a ``float32`` scalar in ``{0.0, 1.0}``.
    """

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        splits_path: str | Path,
        images_dir: str | Path,
        split: str,
        image_size: int = 224,
        aug_prob: float = 0.5,
        aug_severity: float = 0.65,
        base_seed: int = 1234,
        allowed_augs: Sequence[str] = _BENCH_AUG_NAMES,
    ) -> None:
        import torch  # noqa: F401  (validate torch presence up-front)

        splits = load_splits(splits_path)
        if split not in splits:
            raise ValueError(f"unknown split {split!r}; known: {sorted(splits)}")
        wanted = splits[split]

        rows = load_manifest(manifest_path)
        images_dir = Path(images_dir)
        keep: list[_Row] = []
        for r in rows:
            if r.id not in wanted:
                continue
            # Delegate URL → local path to mikha.bench.local_image_path so
            # id-based naming stays consistent with fetch_base.py and never
            # trips over URL-encoded Unicode segments in r.url.
            path = local_image_path(r, images_dir)
            label = 1 if r.image_class == "flood" else 0
            keep.append(_Row(row_id=r.id, image_path=path, label=label))

        if not keep:
            raise ValueError(f"no rows found for split {split!r}")

        for name in allowed_augs:
            if name not in BENCH_TRANSFORMS:
                raise ValueError(f"unknown aug {name!r}; known: {_BENCH_AUG_NAMES}")

        self._rows = keep
        self._image_size = int(image_size)
        self._aug_prob = float(aug_prob)
        self._aug_severity = float(aug_severity)
        self._base_seed = int(base_seed)
        self._allowed_augs = tuple(allowed_augs)
        self._epoch = 0

    # -- optional epoch handle so training loop can bump per-epoch seeding --

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    # -- introspection helpers --

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def labels(self) -> list[int]:
        return [r.label for r in self._rows]

    @property
    def class_counts(self) -> tuple[int, int]:
        n_flood = sum(r.label for r in self._rows)
        return (len(self._rows) - n_flood, n_flood)

    # -- Dataset protocol --

    def __getitem__(self, idx: int):  # noqa: ANN202
        import torch

        row = self._rows[idx]
        image = cv2.imread(str(row.image_path))
        if image is None:
            raise FileNotFoundError(f"could not read {row.image_path}")

        seed = _seed_for(row.row_id, self._epoch, self._base_seed)
        rng = np.random.default_rng(seed)
        if self._aug_prob > 0.0 and rng.random() < self._aug_prob:
            aug_name = self._allowed_augs[int(rng.integers(len(self._allowed_augs)))]
            image = BENCH_TRANSFORMS[aug_name](image, int(seed & 0x7FFFFFFF), self._aug_severity)

        # BGR uint8 → resize → RGB float32 [0,1] → normalize → CHW tensor
        image = cv2.resize(
            image, (self._image_size, self._image_size), interpolation=cv2.INTER_AREA
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        label = torch.tensor(float(row.label), dtype=torch.float32)
        return tensor, label


def compute_pos_weight(dataset: FloodImageDataset) -> float:
    """Return ``n_neg / n_pos`` for ``BCEWithLogitsLoss(pos_weight=…)``.

    Prevents the model from collapsing to the majority class when the
    split is imbalanced. Returns 1.0 if either count is zero (loss falls
    back to unweighted BCE — the dataset is degenerate anyway).
    """
    n_neg, n_pos = dataset.class_counts
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def iter_labels(datasets: Iterable[FloodImageDataset]) -> list[int]:
    """Flatten labels across datasets (utility for sanity checks)."""
    out: list[int] = []
    for d in datasets:
        out.extend(d.labels)
    return out
