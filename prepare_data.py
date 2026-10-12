"""prepare_data.py —— 数据准备：6:2:2 划分 + 生成 annotations.json。

* 把 text/<类>/<同名>.txt 复制到 images/<类>/ 下（txt 很小，几 MB），
  这样 data 目录里 jpg 与同名 txt 就在一起，符合常见的数据组织方式。
* 划分按类别分层（stratified），保证每个类在 train/val/test 里的比例都是 6:2:2，
  否则 100 类里的小类可能整类落进测试集，指标会失去意义。

用法
----
    python prepare_data.py                 # 正常生成
    python prepare_data.py --force         # 覆盖已存在的 annotations.json
    python prepare_data.py --no-sync-text  # 不复制 txt 到 images 目录
    python prepare_data.py --dry-run       # 只打印统计，不写文件
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

# 把本文件所在目录加入 sys.path：
# 这样无论从哪个目录、用 python prepare_data.py 还是 -m 方式启动，
# 都能正确 import 到同目录的 config.py 与 birdmm 包。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from birdmm.utils import Timer, banner, save_json, section


# ---------------------------------------------------------------- 工具
def class_to_prompt(class_name: str) -> str:
    """'001.Black_footed_Albatross' -> 'a photo of a black footed albatross'"""
    name = class_name
    if "." in name:
        prefix, _, rest = name.partition(".")
        if prefix.isdigit():
            name = rest
    name = name.replace("_", " ").strip().lower()
    return f"a photo of a {name}"


def read_captions(txt_path: Path, expected: int = 10) -> list[str]:
    """读一个 txt，返回非空行列表（正常是 10 行）。"""
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]
    caps = [ln for ln in lines if ln]
    return caps


def scan_dataset() -> tuple[list[str], dict[str, list[dict]], dict]:
    """扫描 data/images 与 data/text。

    Returns:
        (类别列表, 每类的样本列表, 统计信息) —— 注意是 **三个** 返回值
    """
    image_root, text_root = config.IMAGE_ROOT, config.TEXT_ROOT
    if not image_root.exists():
        raise FileNotFoundError(
            f"找不到图像目录 {image_root}\n"
            f"请确认数据已解压到 data/ 下（应存在 data/images/ 与 data/text/）"
        )

    classes = sorted([d.name for d in image_root.iterdir() if d.is_dir()])
    if not classes:
        raise FileNotFoundError(f"{image_root} 下没有类别文件夹")

    per_class: dict[str, list[dict]] = {}
    stats = {"no_text": 0, "short_caption": 0, "total_images": 0}

    for cls in classes:
        items = []
        for img in sorted((image_root / cls).glob("*.jpg")):
            stats["total_images"] += 1
            txt = text_root / cls / f"{img.stem}.txt"
            if not txt.exists():
                stats["no_text"] += 1
                continue
            caps = read_captions(txt)
            if len(caps) < config.NUM_CAPTIONS:
                stats["short_caption"] += 1
                if not caps:
                    continue
            items.append({
                "file": img.name,
                "captions": caps,
                "rel_path": f"images/{cls}/{img.name}",
            })
        per_class[cls] = items

    return classes, per_class, stats


def stratified_split(items: list[dict], ratio: tuple[float, float, float] = (0.6, 0.2, 0.2),
                     rng: random.Random | None = None):
    """按 6:2:2 切分单个类别的样本。"""
    rng = rng or random.Random(config.SEED)
    shuffled = list(items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * ratio[0]))
    n_val = int(round(n * ratio[1]))
    # 保证三类都至少有一个样本（样本极少时也不会出现空 split）
    n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, n - 1)
    n_val = max(1, min(n_val, n - n_train - 1)) if n - n_train >= 2 else max(0, n - n_train)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def sync_text_files(classes: list[str]) -> int:
    """把 text/<类>/*.txt 复制到 images/<类>/ 下，让 jpg 与同名 txt 放在一起。"""
    copied = 0
    for cls in classes:
        src_dir = config.TEXT_ROOT / cls
        dst_dir = config.IMAGE_ROOT / cls
        if not src_dir.exists():
            continue
        for txt in src_dir.glob("*.txt"):
            dst = dst_dir / txt.name
            if not dst.exists():
                shutil.copy2(txt, dst)
                copied += 1
    return copied


# ---------------------------------------------------------------- 主流程
def main() -> int:
    parser = argparse.ArgumentParser(description="生成 6:2:2 划分与 annotations.json")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的 annotations.json")
    parser.add_argument("--no-sync-text", action="store_true",
                        help="不把 txt 复制到 images 目录")
    parser.add_argument("--seed", type=int, default=config.SEED, help="随机种子")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    args = parser.parse_args()

    banner("数据准备：6:2:2 划分 + annotations.json")
    print(config.describe())

    if config.ANNOTATIONS.exists() and not args.force:
        print(f"\n[跳过] {config.ANNOTATIONS} 已存在。")
        print("       划分已被冻结，直接训练即可保证各脚本用同一套数据。")
        print("       如需重新划分，请加 --force。")
        return 0

    section("1) 扫描数据集")
    with Timer("扫描"):
        classes, per_class, stats = scan_dataset()

    print(f"类别数        : {len(classes)}")
    print(f"图像总数      : {stats['total_images']}")
    print(f"缺少同名 txt  : {stats['no_text']}  （这些图无法参与文本/多模态任务，已剔除）")
    print(f"描述不足 10 条: {stats['short_caption']}")

    usable = sum(len(v) for v in per_class.values())
    print(f"可用样本      : {usable}")
    counts = [len(v) for v in per_class.values()]
    print(f"每类样本数    : min={min(counts)}  max={max(counts)}  avg={sum(counts) / len(counts):.1f}")

    section("2) 按类别分层做 6:2:2 划分")
    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for label, cls in enumerate(classes):
        items = per_class[cls]
        tr, va, te = stratified_split(items, config.SPLIT_RATIO, rng)
        for split_name, group in (("train", tr), ("val", va), ("test", te)):
            for it in group:
                splits[split_name].append({
                    "image_path": it["rel_path"],
                    "captions": it["captions"],
                    "label": label,
                    "class_name": cls,
                })

    total = sum(len(v) for v in splits.values())
    for name in ("train", "val", "test"):
        n = len(splits[name])
        print(f"{name:<6}: {n:>5} 张  ({n / total:.1%})")

    # 校验：每个类在 train 里都必须出现，否则训练时该类永远学不到
    train_labels = {r["label"] for r in splits["train"]}
    val_labels = {r["label"] for r in splits["val"]}
    test_labels = {r["label"] for r in splits["test"]}
    print(f"train 覆盖类别: {len(train_labels)}/{len(classes)}")
    print(f"val   覆盖类别: {len(val_labels)}/{len(classes)}")
    print(f"test  覆盖类别: {len(test_labels)}/{len(classes)}")
    if len(train_labels) < len(classes):
        missing = sorted(set(range(len(classes))) - train_labels)
        print(f"[警告] 以下类别在 train 中一个样本都没有：{[classes[i] for i in missing][:10]}")

    annotation = {
        "meta": {
            "source": "CUB_200_subset (100 classes)",
            "num_classes": len(classes),
            "num_captions": config.NUM_CAPTIONS,
            "split_ratio": list(config.SPLIT_RATIO),
            "seed": args.seed,
            "total_images": total,
            "excluded_no_text": stats["no_text"],
            "created_by": "prepare_data.py",
        },
        "classes": classes,
        "class_prompts": {cls: class_to_prompt(cls) for cls in classes},
        "splits": splits,
    }

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件。")
        return 0

    section("3) 写出 annotations.json")
    save_json(annotation, config.ANNOTATIONS)
    size_mb = config.ANNOTATIONS.stat().st_size / 1024 / 1024
    print(f"已写出: {config.ANNOTATIONS}  ({size_mb:.1f} MB)")

    if not args.no_sync_text:
        section("4) 把 txt 复制到 images/<类>/ 下（让 jpg 与同名 txt 放在一起）")
        copied = sync_text_files(classes)
        print(f"复制了 {copied} 个 txt 文件到 {config.IMAGE_ROOT}")
        sample_cls = classes[0]
        sample = next((config.IMAGE_ROOT / sample_cls).glob("*.jpg"), None)
        if sample is not None:
            print(f"示例: {sample}")
            print(f"      {sample.with_suffix('.txt')}")

    section("完成")
    print(f"下一步：python 01_image_only.py    （图像单模态）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
