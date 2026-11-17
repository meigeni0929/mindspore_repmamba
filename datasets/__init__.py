"""Dataset package; implemented after the inference correctness baseline."""
"""Deterministic paired-image data pipeline for x4 super resolution."""

from pathlib import Path
import random

import numpy as np
from PIL import Image
import mindspore.dataset as ds


def paired_paths(root, lr_dir="LR", gt_dir="HR"):
    root = Path(root)
    lr_files = {p.name: p for p in (root / lr_dir).iterdir() if p.is_file()}
    gt_files = {p.name: p for p in (root / gt_dir).iterdir() if p.is_file()}
    if set(lr_files) != set(gt_files):
        raise ValueError("LR/GT filenames do not match: missing_lr={}, missing_gt={}".format(
            sorted(set(gt_files) - set(lr_files)), sorted(set(lr_files) - set(gt_files))))
    if not lr_files:
        raise ValueError("no paired images found")
    return [(lr_files[name], gt_files[name]) for name in sorted(lr_files)]


def decode_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def transform_pair(lr, gt, scale=4, patch_size=None, crop=(0, 0),
                   hflip=False, vflip=False, rotation=0):
    """Apply identical deterministic geometry; return FP32 CHW in [0,1]."""
    if gt.shape[0] != lr.shape[0] * scale or gt.shape[1] != lr.shape[1] * scale:
        raise ValueError("GT dimensions must be exactly scale times LR dimensions")
    if patch_size is not None:
        top, left = crop
        if top < 0 or left < 0 or top + patch_size > lr.shape[0] or left + patch_size > lr.shape[1]:
            raise ValueError("crop is outside LR image")
        lr = lr[top:top + patch_size, left:left + patch_size]
        gt = gt[top * scale:(top + patch_size) * scale,
                left * scale:(left + patch_size) * scale]
    if hflip:
        lr, gt = lr[:, ::-1], gt[:, ::-1]
    if vflip:
        lr, gt = lr[::-1], gt[::-1]
    if rotation not in (0, 1, 2, 3):
        raise ValueError("rotation must be 0, 1, 2 or 3 quarter turns")
    if rotation:
        lr, gt = np.rot90(lr, rotation), np.rot90(gt, rotation)
    return (np.ascontiguousarray(lr.transpose(2, 0, 1), dtype=np.float32),
            np.ascontiguousarray(gt.transpose(2, 0, 1), dtype=np.float32))


class PairedImageSource:
    def __init__(self, root, scale=4, patch_size=None, training=False, seed=66,
                 lr_dir="LR", gt_dir="HR"):
        self.pairs = paired_paths(root, lr_dir, gt_dir)
        self.scale, self.patch_size, self.training, self.seed = scale, patch_size, training, seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        lr_path, gt_path = self.pairs[index]
        lr, gt = decode_rgb(lr_path), decode_rgb(gt_path)
        rng = random.Random(self.seed + index)
        crop = (0, 0)
        if self.patch_size is not None:
            crop = (rng.randrange(lr.shape[0] - self.patch_size + 1),
                    rng.randrange(lr.shape[1] - self.patch_size + 1))
        return transform_pair(lr, gt, self.scale, self.patch_size, crop,
                              rng.choice((False, True)) if self.training else False,
                              rng.choice((False, True)) if self.training else False,
                              rng.randrange(4) if self.training else 0)


def create_dataset(root, scale=4, patch_size=None, training=False, seed=66,
                   batch_size=1, shuffle=None, num_workers=1, drop_remainder=False,
                   lr_dir="LR", gt_dir="HR"):
    source = PairedImageSource(root, scale, patch_size, training, seed, lr_dir, gt_dir)
    dataset = ds.GeneratorDataset(source, ["lr", "gt"], shuffle=training if shuffle is None else shuffle,
                                  num_parallel_workers=num_workers)
    return dataset.batch(batch_size, drop_remainder=drop_remainder)


__all__ = ["paired_paths", "decode_rgb", "transform_pair", "PairedImageSource", "create_dataset"]
