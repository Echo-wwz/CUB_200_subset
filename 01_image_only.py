"""01_image_only.py —— 阶段 2a 图像单模态训练（Image-Only）。

本脚本做的事
------------
1. 用 data/annotations.json 的 6:2:2 划分训练 ResNet18 / MobileNetV2
2. 在 test 上报告 Accuracy / Precision / Recall / F1 / Top-5
3. 落盘：训练曲线、混淆矩阵、错误分析（最易混淆的类别对 + 具体错例）

----
    python 01_image_only.py                          # 默认 resnet18（冻结主干）
    python 01_image_only.py --backbone mobilenet_v2  # 换主干
    python 01_image_only.py --no-freeze --epochs 10  # 解冻主干做全量微调
    python 01_image_only.py --epochs 2 --limit-batches 20   # 冒烟测试
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 先检查依赖：缺 torch 时给出可照做的提示，而不是一句 ModuleNotFoundError
from birdmm import require

require("torch", "torchvision", "numpy", "PIL")

import numpy as np
import torch
import torch.nn as nn

import config
from birdmm.data import Annotations, make_image_loaders
from birdmm.engine import evaluate, train_model
from birdmm.image_encoder import ImageClassifier
from birdmm.metrics import format_metrics, get_confusion_matrix, per_class_accuracy, top_confusions
from birdmm.utils import (HAS_MPL, banner, human_num, load_json, log_to_file, save_json,
                          section, set_seed, to_builtin, count_parameters)
from birdmm import viz


def parse_args():
    p = argparse.ArgumentParser(description="图像单模态训练（ResNet18 / MobileNetV2）")
    p.add_argument("--backbone", default=config.IMG_BACKBONE,
                   choices=["resnet18", "resnet34", "resnet50", "mobilenet_v2", "mobilenet_v3_small"],
                   help="图像主干网络")
    p.add_argument("--epochs", type=int, default=config.IMG_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.IMG_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=None, help="默认：冻结=1e-3，解冻=1e-4")
    p.add_argument("--no-freeze", action="store_true", help="解冻主干做全量微调")
    p.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    p.add_argument("--limit-batches", type=int, default=None,
                   help="每个 epoch 只跑前 N 个 batch（冒烟测试用）")
    p.add_argument("--no-pretrained", action="store_true", help="不用 ImageNet 预训练权重")
    p.add_argument("--tag", default=None, help="结果文件名后缀，默认用 backbone 名")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tag = args.tag or args.backbone
    tee = log_to_file(config.LOG_ROOT / f"01_image_only_{tag}.log")

    try:
        set_seed(config.SEED)
        device = config.get_device(args.device)
        freeze = not args.no_freeze
        lr = args.lr if args.lr is not None else (config.IMG_LR if freeze else config.IMG_LR_FINETUNE)

        banner("阶段 2a  图像单模态训练（Image-Only）")
        print(config.describe())
        print(f"主干={args.backbone}  冻结主干={freeze}  lr={lr}  epochs={args.epochs}  "
              f"batch={args.batch_size}  device={device}")

        # ---------------------------------------------------------- 数据
        section("1) 加载数据（使用 annotations.json 里冻结的 6:2:2 划分）")
        ann = Annotations()
        print(ann.summary())
        sets, loaders = make_image_loaders(ann, batch_size=args.batch_size)
        print(f"类别数: {ann.num_classes}")
        sample = sets["train"][0]
        print(f"单张图像张量形状: {tuple(sample['pixel_values'].shape)}  "
              f"（[3, {config.IMAGE_SIZE}, {config.IMAGE_SIZE}]）")
        print(f"对应标签        : {int(sample['label'].item())} -> {ann.classes[int(sample['label'])]}")

        # ---------------------------------------------------------- 模型
        section("2) 构建模型")
        model = ImageClassifier(
            backbone=args.backbone,
            num_classes=ann.num_classes,
            pretrained=not args.no_pretrained,
            freeze=freeze,
            dropout=0.2,
        )
        total, trainable = count_parameters(model)
        print(model)
        print(f"总参数量     : {human_num(total)}")
        print(f"可训练参数量 : {human_num(trainable)}  ({trainable / total:.1%})")

        # ---------------------------------------------------------- 训练
        section("3) 训练")
        ckpt = config.CKPT_ROOT / f"01_image_{tag}.pt"
        result = train_model(
            model, loaders["train"], loaders["val"], device,
            epochs=args.epochs, lr=lr, num_classes=ann.num_classes,
            max_train_batches=args.limit_batches, max_eval_batches=args.limit_batches,
            tag=f"[img:{args.backbone}]", save_path=ckpt,
        )

        # ---------------------------------------------------------- 测试
        section("4) 测试集评估")
        criterion = nn.CrossEntropyLoss()
        test_metrics = evaluate(model, loaders["test"], criterion, device,
                                num_classes=ann.num_classes, collect=True)
        y_true, y_pred = test_metrics.pop("_y_true"), test_metrics.pop("_y_pred")
        probs = test_metrics.pop("_probs")
        test_metrics.pop("_logits", None)

        print(format_metrics(test_metrics))

        # ---------------------------------------------------------- 保存
        section("5) 保存结果")
        out = {
            "phase": "stage-2a image-only",
            "backbone": args.backbone,
            "freeze_backbone": freeze,
            "epochs": args.epochs,
            "lr": lr,
            "batch_size": args.batch_size,
            "pretrained": not args.no_pretrained,
            "params_total": total,
            "params_trainable": trainable,
            "best_epoch": result["best_epoch"],
            "train_time_sec": round(result["train_time"], 1),
            "history": result["history"],
            "test": {k: v for k, v in test_metrics.items()},
        }
        save_json(to_builtin(out), config.RESULT_ROOT / f"01_image_only_{tag}.json")
        print(f"指标 -> {config.RESULT_ROOT / f'01_image_only_{tag}.json'}")
        print(f"权重 -> {ckpt}")

        # ---------------------------------------------------------- 可视化
        if HAS_MPL:
            section("6) 可视化")
            viz.plot_train_curve(result["history"],
                                 config.FIG_ROOT / f"train_curve_image_{tag}.png",
                                 title=f"图像单模态 {args.backbone}")
            cm = get_confusion_matrix(y_true, y_pred, ann.num_classes)
            viz.plot_confusion_matrix(cm, config.FIG_ROOT / f"confusion_matrix_image_{tag}.png",
                                      ann.classes, title=f"图像单模态 {args.backbone} 测试集")
        else:
            print("\n[跳过可视化] 未安装 matplotlib")

        # ---------------------------------------------------------- 错误分析
        section("7) 错误分析：最容易混淆的类别")
        conf = top_confusions(y_true, y_pred, ann.classes, ann.num_classes, topn=10)
        print(f"{'真实类别':<34}{'被预测为':<34}{'次数':>6}")
        for c in conf:
            print(f"{c['true_class']:<34}{c['pred_class']:<34}{c['count']:>6}")

        accs = per_class_accuracy(y_true, y_pred, ann.num_classes)
        order = np.argsort(accs)
        print("\n准确率最低的 10 个类别（这些类最需要看文本描述帮忙）：")
        for i in order[:10]:
            print(f"  {ann.classes[i]:<34} acc={accs[i]:.3f}")

        # 具体错例（含模型给错答案时的置信度）
        wrong = np.where(y_pred != y_true)[0]
        print(f"\n错例总数: {len(wrong)} / {len(y_true)}")
        if len(wrong):
            print(f"{'真实':<28}{'预测':<28}{'置信度':>8}")
            for idx in wrong[np.argsort(-probs[wrong].max(axis=1))][:8]:
                print(f"{ann.classes[y_true[idx]][:26]:<28}"
                      f"{ann.classes[y_pred[idx]][:26]:<28}{probs[idx].max():>8.3f}")

        save_json(to_builtin({
            "top_confusions": conf,
            "lowest_accuracy_classes": [
                {"class": ann.classes[i], "accuracy": float(accs[i])} for i in order[:15]],
            "num_errors": int(len(wrong)),
        }), config.RESULT_ROOT / f"01_image_only_{tag}_errors.json")

        section("完成")
        print(f"测试集准确率: {test_metrics['accuracy']:.4f}   "
              f"宏平均 F1: {test_metrics['f1']:.4f}")
        print("下一步: python 02_text_only.py   （文本单模态）")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    sys.exit(main())
