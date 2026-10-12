"""data.py —— 三个 Dataset + 图像预处理。

    ImageDataset       图像单模态   -> (image, label)
    TextDataset        文本单模态   -> (input_ids, attention_mask, label)
    MultimodalDataset  多模态       -> (image, input_ids, attention_mask, label)

三者都读同一份 data/annotations.json，因此 **train/val/test 划分完全一致**，
单模态与多模态的结果才可以直接比较（这是消融实验成立的前提）。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
from .utils import HAS_TQDM


# ------------------------------------------------------------------ 预处理
def build_image_transform(train: bool, size: int | None = None):
    """返回 torchvision transform。

    训练集：RandomResizedCrop + 翻转 + 颜色抖动（数据增强）
    验证/测试：Resize(256) + CenterCrop(224)（只做确定性变换）
    """
    from torchvision import transforms

    size = size or config.IMAGE_SIZE
    mean, std = config.IMAGENET_MEAN, config.IMAGENET_STD

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ------------------------------------------------------------------ 标注加载
class Annotations:
    """annotations.json 的轻量包装。"""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.ANNOTATIONS)
        if not self.path.exists():
            raise FileNotFoundError(
                f"找不到标注文件 {self.path}\n"
                f"请先运行：python prepare_data.py"
            )
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.meta: Dict = raw.get("meta", {})
        self.classes: List[str] = raw["classes"]
        self.class_prompts: Dict[str, str] = raw.get("class_prompts", {})
        self.splits: Dict[str, List[dict]] = raw["splits"]

    def __len__(self) -> int:
        return len(self.classes)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def records(self, split: str) -> List[dict]:
        return self.splits[split]

    def summary(self) -> str:
        lines = [f"标注文件   : {self.path}",
                 f"类别数     : {self.num_classes}",
                 f"每类描述数 : {self.meta.get('num_captions', config.NUM_CAPTIONS)}"]
        for split in ("train", "val", "test"):
            recs = self.splits.get(split, [])
            lines.append(f"{split:<10} : {len(recs):>5} 张图")
        return "\n".join(lines)


# ------------------------------------------------------------------ Dataset
class _BaseBirdDataset(Dataset):
    def __init__(self, records: List[dict], classes: List[str]):
        self.records = records
        self.classes = classes

    def __len__(self) -> int:
        return len(self.records)


class ImageDataset(_BaseBirdDataset):
    """图像单模态数据集。

    返回 dict（与 TextDataset / MultimodalDataset 保持同一套键名约定），
    这样 engine.forward_batch 可以用统一逻辑喂给模型。
    """

    def __init__(self, records, classes, transform=None):
        super().__init__(records, classes)
        self.transform = transform or build_image_transform(train=False)

    def __getitem__(self, index: int):
        rec = self.records[index]
        img = Image.open(config.DATA_ROOT / rec["image_path"]).convert("RGB")
        return {
            "pixel_values": self.transform(img),
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
        }


class TextDataset(_BaseBirdDataset):
    """文本单模态数据集：每张图有 10 条描述，每次随机取一条。

    训练时随机采样等价于对描述做数据增强；验证/测试时固定取第 0 条，
    保证每次评估用的是同一批文本（否则指标会抖动，无法比较）。
    """

    def __init__(self, records, classes, tokenizer, max_length: int | None = None,
                 random_caption: bool = True):
        super().__init__(records, classes)
        self.tokenizer = tokenizer
        self.max_length = max_length or config.TEXT_MAX_LEN
        self.random_caption = random_caption

    def __getitem__(self, index: int):
        rec = self.records[index]
        captions = rec["captions"]
        if self.random_caption:
            text = random.choice(captions)
        else:
            text = captions[0]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
        }


class MultimodalDataset(_BaseBirdDataset):
    """多模态数据集：同一张图 + 它的一条描述，同时给出。

    额外返回 index，方便把预测结果对应回原图（错误分析要用）。
    """

    def __init__(self, records, classes, tokenizer, transform=None, max_length: int | None = None,
                 random_caption: bool = True):
        super().__init__(records, classes)
        self.tokenizer = tokenizer
        self.transform = transform or build_image_transform(train=False)
        self.max_length = max_length or config.TEXT_MAX_LEN
        self.random_caption = random_caption

    def __getitem__(self, index: int):
        rec = self.records[index]
        image = Image.open(config.DATA_ROOT / rec["image_path"]).convert("RGB")
        pixel = self.transform(image)

        captions = rec["captions"]
        text = random.choice(captions) if self.random_caption else captions[0]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
            "caption": text,
        }


# ------------------------------------------------------------------ DataLoader
def _loader_kwargs(batch_size: int, shuffle: bool, num_workers: int) -> dict:
    return dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def make_image_loaders(ann: Annotations, batch_size: int | None = None,
                      num_workers: int | None = None):
    batch_size = batch_size or config.IMG_BATCH_SIZE
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers
    sets = {
        "train": ImageDataset(ann.records("train"), ann.classes, build_image_transform(True)),
        "val": ImageDataset(ann.records("val"), ann.classes, build_image_transform(False)),
        "test": ImageDataset(ann.records("test"), ann.classes, build_image_transform(False)),
    }
    loaders = {
        split: DataLoader(ds, **_loader_kwargs(batch_size, split == "train", num_workers))
        for split, ds in sets.items()
    }
    return sets, loaders


def make_text_loaders(ann: Annotations, tokenizer, batch_size: int | None = None,
                     num_workers: int | None = None):
    batch_size = batch_size or config.TXT_BATCH_SIZE
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers
    sets = {
        "train": TextDataset(ann.records("train"), ann.classes, tokenizer, random_caption=True),
        "val": TextDataset(ann.records("val"), ann.classes, tokenizer, random_caption=False),
        "test": TextDataset(ann.records("test"), ann.classes, tokenizer, random_caption=False),
    }
    loaders = {
        split: DataLoader(ds, **_loader_kwargs(batch_size, split == "train", num_workers))
        for split, ds in sets.items()
    }
    return sets, loaders


def make_multimodal_loaders(ann: Annotations, tokenizer, batch_size: int | None = None,
                           num_workers: int | None = None):
    batch_size = batch_size or config.FUSION_BATCH_SIZE
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers
    sets = {
        "train": MultimodalDataset(ann.records("train"), ann.classes, tokenizer,
                                   build_image_transform(True), random_caption=True),
        "val": MultimodalDataset(ann.records("val"), ann.classes, tokenizer,
                                 build_image_transform(False), random_caption=False),
        "test": MultimodalDataset(ann.records("test"), ann.classes, tokenizer,
                                  build_image_transform(False), random_caption=False),
    }
    loaders = {
        split: DataLoader(ds, **_loader_kwargs(batch_size, split == "train", num_workers))
        for split, ds in sets.items()
    }
    return sets, loaders
