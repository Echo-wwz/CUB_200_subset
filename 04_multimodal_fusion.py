"""04_multimodal_fusion.py —— 阶段 4 多模态融合 + 消融实验（本项目的核心）。

四种融合模块的接口完全一致：
    forward(image_feat, text_feat, attention_mask) -> {"logits": ...}
"换融合方式只需要改配置里的 fusion: 一个字段 —— 这也是消融实验能做得干净的前提。"

本脚本就是这句话的可运行版本：
    --fusion early / late / attention / gated / all

四种融合
--------
① Early      [图像512, 文本768] -> concat -> MLP -> 类别
② Late       图像头 -> p_img，文本头 -> p_text，w·p_text + (1-w)·p_img
③ Attention  图像 token[49,512] 作 Query，文本 token[L,768] 作 Key/Value -> 池化 -> 类别
④ Gated      a = σ(W[img;txt])，fusion = a·img + (1-a)·txt

额外做的分析
------------
* 模态缺失实验：Late / Gated 天然支持"只给图像"或"只给文本"，正好回答
  "两个模态各自缺什么"
* 注意力可视化：Attention 融合可以导出权重，看"模型在看哪个词"
* 门控分布：Gated 融合的 gate 值分布，看模型更信谁
* 与单模态基线对比：自动读取 01/02 的结果 json 放在同一张表里

用法
----
    python 04_multimodal_fusion.py --fusion all                 # 跑完四种融合做消融
    python 04_multimodal_fusion.py --fusion attention           # 只跑推荐的跨模态注意力
    python 04_multimodal_fusion.py --fusion all --epochs 2 --limit-batches 20   # 冒烟
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 先检查依赖：缺 torch 时给出可照做的提示，而不是一句 ModuleNotFoundError
from birdmm import require

require('torch', 'torchvision', 'transformers', 'numpy', 'PIL')

import numpy as np
import torch
import torch.nn as nn

import config
from birdmm.data import Annotations, make_multimodal_loaders
from birdmm.engine import evaluate, train_model
from birdmm.fusion import FUSION_DESCRIPTIONS, FUSION_TYPES, build_fusion, normalize_fusion_type
from birdmm.image_encoder import ImageEncoder
from birdmm.metrics import format_metrics, get_confusion_matrix, per_class_accuracy, top_confusions
from birdmm.text_encoder import TextEncoder, load_tokenizer
from birdmm.utils import (HAS_MPL, banner, count_parameters, human_num, log_to_file, save_json,
                          section, set_seed, to_builtin)
from birdmm import viz


# ==================================================================== 模型
class MultimodalClassifier(nn.Module):
    """图像编码器 + 文本编码器 + 融合模块。

    两个编码器的输出维度不同（512 / 768），由融合模块内部的投影层统一到
    公共维度 d（config.FUSION_DIM），这样四种融合方式的参数量才可比。
    """

    def __init__(self, fusion_type: str = "attention", num_classes: int = 100,
                 image_backbone: str = "resnet18", image_pretrained: bool = True,
                 image_freeze: bool = True, text_model: str = "bert-base-uncased",
                 text_freeze: bool = True, text_pooling: str = "mean",
                 fusion_dim: int | None = 256, hidden_dim: int = 512, dropout: float = 0.3,
                 num_heads: int = 4):
        super().__init__()
        self.fusion_type = normalize_fusion_type(fusion_type)
        self.image_encoder = ImageEncoder(image_backbone, pretrained=image_pretrained,
                                          freeze=image_freeze)
        self.text_encoder = TextEncoder(text_model, pretrained=True, freeze=text_freeze,
                                        pooling=text_pooling)
        self.fusion = build_fusion(
            self.fusion_type,
            image_dim=self.image_encoder.out_dim,
            text_dim=self.text_encoder.out_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            common_dim=fusion_dim,
            num_heads=num_heads,
        )
        # Attention 融合需要 token 序列；其余三种只需要全局向量
        self.needs_tokens = self.fusion_type == "attention"

    def encode(self, pixel_values, input_ids, attention_mask):
        """分别编码两个模态，返回 (image_feat, text_feat, image_tokens, text_tokens)。"""
        image_tokens = text_tokens = None
        if self.needs_tokens:
            image_tokens = self.image_encoder(pixel_values, return_tokens=True)
            text_feat, text_tokens = self.text_encoder(input_ids, attention_mask,
                                                       return_tokens=True)
            image_feat = image_tokens.mean(dim=1)          # 仅用于统计/可视化
        else:
            image_feat = self.image_encoder(pixel_values)
            text_feat = self.text_encoder(input_ids, attention_mask)
        return image_feat, text_feat, image_tokens, text_tokens

    def forward(self, pixel_values, input_ids, attention_mask, return_attn: bool | None = None):
        image_feat, text_feat, image_tokens, text_tokens = self.encode(
            pixel_values, input_ids, attention_mask)
        return self.fusion(image_feat, text_feat, attention_mask=attention_mask,
                           text_tokens=text_tokens, image_tokens=image_tokens,
                           return_attn=return_attn)


# ==================================================================== 评估
@torch.no_grad()
def evaluate_multimodal(model, loader, device, num_classes: int, collect: bool = False):
    """多模态评估；额外收集 gate / 注意力权重，供可视化用。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_logits, all_labels = [], []
    gates, attn_store = [], None
    records_meta = {"captions": [], "index": []}

    for batch in loader:
        pixel = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(pixel, ids, mask)
        logits = out["logits"]

        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(batch["label"].numpy())
        if "gate" in out:
            gates.append(out["gate"].float().cpu().numpy())
        if "attn_weights" in out and attn_store is None:
            attn_store = out["attn_weights"].float().cpu().numpy()   # 只留第一个 batch
        if collect:
            records_meta["captions"].extend(batch["caption"])
            records_meta["index"].extend(batch["index"].numpy().tolist())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = logits.argmax(axis=1)

    from birdmm.metrics import compute_metrics
    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes, logits=logits)
    result = {"metrics": metrics, "y_true": y_true, "y_pred": y_pred,
              "logits": logits, "loss": float(criterion(
                  torch.tensor(logits), torch.tensor(y_true)).item())}
    if gates:
        result["gates"] = np.concatenate(gates, axis=0)
    if attn_store is not None:
        result["attn_weights"] = attn_store
    if collect:
        result["meta"] = records_meta
    return result


@torch.no_grad()
def evaluate_single_modality(model, loader, device, num_classes: int, modality: str):
    """模态缺失实验：只用图像或只用文本走融合模块（仅 Late / Gated 支持）。"""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        pixel = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        image_feat, text_feat, _, _ = model.encode(pixel, ids, mask)
        kwargs = {"image_only": True} if modality == "image" else {"text_only": True}
        out = model.fusion(image_feat, text_feat, attention_mask=mask, **kwargs)
        all_logits.append(out["logits"].float().cpu().numpy())
        all_labels.append(batch["label"].numpy())

    from birdmm.metrics import compute_metrics
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return compute_metrics(y_true, logits.argmax(axis=1), num_classes=num_classes, logits=logits)


# ==================================================================== 单次实验
def run_one_fusion(fusion_type: str, ann: Annotations, loaders, device, args) -> dict:
    tag = fusion_type
    section(f"融合方式：{fusion_type}  —— {FUSION_DESCRIPTIONS[fusion_type]}")

    set_seed(config.SEED)
    model = MultimodalClassifier(
        fusion_type=fusion_type,
        num_classes=ann.num_classes,
        image_backbone=args.image_backbone,
        image_pretrained=not args.no_pretrained,
        image_freeze=not args.no_freeze,
        text_model=args.text_model,
        text_freeze=not args.no_freeze_text,
        text_pooling=args.text_pooling,
        fusion_dim=args.fusion_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_heads=args.num_heads,
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"总参数量={human_num(total)}  可训练={human_num(trainable)} ({trainable / total:.1%})")
    print(f"图像编码器输出={model.image_encoder.out_dim}  文本编码器输出={model.text_encoder.out_dim}"
          f"  公共维度={args.fusion_dim}")

    ckpt = config.CKPT_ROOT / f"04_fusion_{tag}.pt"
    result = train_model(
        model, loaders["train"], loaders["val"], device,
        epochs=args.epochs, lr=args.lr, num_classes=ann.num_classes,
        max_train_batches=args.limit_batches, max_eval_batches=args.limit_batches,
        tag=f"[{tag}]", save_path=ckpt,
    )

    section(f"[{tag}] 测试集评估")
    test = evaluate_multimodal(model, loaders["test"], device, ann.num_classes, collect=True)
    metrics = test["metrics"]
    print(format_metrics(metrics))

    extra: dict = {}
    # Late 融合的可学习权重 w —— 这是它"可解释"的体现
    weight_prop = getattr(model.fusion, "weight", None)
    if isinstance(weight_prop, torch.Tensor):
        w = float(weight_prop.detach().cpu())
        extra["fusion_weight_w"] = w
        print(f"\n可学习融合权重 w = {w:.4f}  "
              f"（w 是文本的权重，{w:.1%} 文本 + {1 - w:.1%} 图像）")

    # 模态缺失实验
    if fusion_type in ("late", "gated"):
        section(f"[{tag}] 模态缺失实验（两个模态各自缺什么）")
        extra["missing_modality"] = {}
        for modality, label in (("image", "只有图像"), ("text", "只有文本")):
            met = evaluate_single_modality(model, loaders["test"], device, ann.num_classes, modality)
            extra["missing_modality"][modality] = met
            print(f"  {label:<8} acc={met['accuracy']:.4f}  macro-F1={met['f1']:.4f}")
        print(f"  {'双模态':<8} acc={metrics['accuracy']:.4f}  macro-F1={metrics['f1']:.4f}  ← 融合后")
        gain = metrics["accuracy"] - max(extra["missing_modality"]["image"]["accuracy"],
                                        extra["missing_modality"]["text"]["accuracy"])
        print(f"  => 融合带来的增益: {gain:+.4f}")
        extra["fusion_gain"] = float(gain)
    else:
        print(f"\n[说明] {fusion_type} 融合的接口要求两个模态都在场，")
        print(f"       无法直接做模态缺失实验 —— 这正是 Late Fusion 的独特优势。")

    # 保存
    out = {
        "phase": "stage-4 multimodal fusion",
        "fusion_type": fusion_type,
        "description": FUSION_DESCRIPTIONS[fusion_type],
        "image_backbone": args.image_backbone,
        "image_freeze": not args.no_freeze,
        "text_model": args.text_model,
        "text_freeze": not args.no_freeze_text,
        "text_pooling": args.text_pooling,
        "fusion_dim": args.fusion_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "params_total": total,
        "params_trainable": trainable,
        "best_epoch": result["best_epoch"],
        "train_time_sec": round(result["train_time"], 1),
        "history": result["history"],
        "test": metrics,
        **extra,
    }
    save_json(to_builtin(out), config.RESULT_ROOT / f"04_fusion_{tag}.json")
    print(f"\n指标 -> {config.RESULT_ROOT / f'04_fusion_{tag}.json'}")
    print(f"权重 -> {ckpt}")

    # ---------------------------------------------------------- 可视化
    if HAS_MPL:
        viz.plot_train_curve(result["history"],
                             config.FIG_ROOT / f"train_curve_fusion_{tag}.png",
                             title=f"融合方式: {fusion_type}")
        cm = get_confusion_matrix(test["y_true"], test["y_pred"], ann.num_classes)
        viz.plot_confusion_matrix(cm, config.FIG_ROOT / f"confusion_matrix_fusion_{tag}.png",
                                  ann.classes, title=f"多模态融合({fusion_type}) 测试集")

        if "gates" in test:
            viz.plot_gate_distribution(test["gates"], test["y_true"],
                                       config.FIG_ROOT / f"gate_distribution_{tag}.png",
                                       ann.classes)

        if "attn_weights" in test:
            _plot_attention_example(model, loaders, device, ann, tag)

    # ---------------------------------------------------------- 错误分析
    section(f"[{tag}] 错误分析")
    conf = top_confusions(test["y_true"], test["y_pred"], ann.classes, ann.num_classes, topn=8)
    print(f"{'真实类别':<34}{'被预测为':<34}{'次数':>6}")
    for c in conf:
        print(f"{c['true_class']:<34}{c['pred_class']:<34}{c['count']:>6}")
    save_json(to_builtin({"top_confusions": conf}), 
              config.RESULT_ROOT / f"04_fusion_{tag}_errors.json")

    return out


def _plot_attention_example(model, loaders, device, ann, tag: str):
    """取一个测试样本，把跨模态注意力画出来（模型在看哪个词）。"""
    from birdmm.text_encoder import load_tokenizer as _lt

    tokenizer = _lt(config.BERT_NAME)
    batch = next(iter(loaders["test"]))
    pixel = batch["pixel_values"][:1].to(device)
    ids = batch["input_ids"][:1].to(device)
    mask = batch["attention_mask"][:1].to(device)

    model.eval()
    with torch.no_grad():
        image_feat, text_feat, image_tokens, text_tokens = model.encode(pixel, ids, mask)
        out = model.fusion(image_feat, text_feat, attention_mask=mask,
                           text_tokens=text_tokens, image_tokens=image_tokens,
                           return_attn=True)

    attn = out.get("attn_weights")
    if attn is None:
        return
    attn = attn[0].cpu().numpy()                       # [heads, 49, L]
    toks = tokenizer.convert_ids_to_tokens(ids[0].cpu().tolist())
    real = int(mask[0].sum().item())
    toks = toks[:real]                                 # 去掉 [PAD]
    attn = attn[:, :, :real]

    caption = batch["caption"][0]
    print(f"\n  注意力样本描述: {caption!r}")
    viz.plot_attention(attn, toks, config.FIG_ROOT / f"attention_{tag}.png",
                       title=f"跨模态注意力（{tag}）—— 图像 49 个位置各自在看哪个词\n描述: {caption[:60]}")

    # 多 head 对比：不同 head 关注不同的词
    n_heads = attn.shape[0]
    if n_heads > 1 and HAS_MPL:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, n_heads, figsize=(3.2 * n_heads, 3.4), squeeze=False)
        for h in range(n_heads):
            ax = axes[0][h]
            per_tok = attn[h].mean(axis=0)
            ax.bar(range(len(toks)), per_tok, color="#4C72B0")
            ax.set_xticks(range(len(toks)))
            ax.set_xticklabels(toks, rotation=90, fontsize=7)
            ax.set_title(f"head {h}", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("不同注意力头关注的词不同（每个 head 的 49 个图像位置平均注意力）", fontsize=11)
        viz._save(fig, config.FIG_ROOT / f"attention_heads_{tag}.png")
    print(f"[图] 注意力可视化 -> {config.FIG_ROOT / f'attention_{tag}.png'}")


# ==================================================================== 主流程
def parse_args():
    p = argparse.ArgumentParser(description="多模态融合与消融实验")
    p.add_argument("--fusion", default="all",
                   choices=list(FUSION_TYPES) + ["all"],
                   help="融合方式；all = 依次跑完四种做消融")
    p.add_argument("--epochs", type=int, default=config.FUSION_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.FUSION_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.FUSION_LR)
    p.add_argument("--image-backbone", default=config.IMG_BACKBONE)
    p.add_argument("--text-model", default=config.BERT_NAME)
    p.add_argument("--text-pooling", default=config.TXT_POOLING, choices=["cls", "mean"])
    p.add_argument("--fusion-dim", type=int, default=config.FUSION_DIM,
                   help="两个模态统一到的公共维度 d")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--no-freeze", action="store_true", help="解冻图像主干（慢，但可能更好）")
    p.add_argument("--no-freeze-text", action="store_true", help="解冻 BERT")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config.SEED = args.seed
    tee = log_to_file(config.LOG_ROOT / "04_multimodal_fusion.log")

    try:
        set_seed(args.seed)
        device = config.get_device(args.device)

        banner("阶段 4  多模态融合（四种融合方式 + 消融实验）")
        print(config.describe())
        print(f"device={device}  epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}")
        print(f"公共维度 d={args.fusion_dim}  hidden={args.hidden_dim}  dropout={args.dropout}")
        print(f"冻结图像主干={not args.no_freeze}  冻结BERT={not args.no_freeze_text}")

        section("0) 数据（图文配对：同一张图 + 它的一条描述）")
        ann = Annotations()
        print(ann.summary())
        tokenizer = load_tokenizer(args.text_model)
        sets, loaders = make_multimodal_loaders(ann, tokenizer, batch_size=args.batch_size)
        sample = sets["train"][0]
        print(f"样本键: {list(sample.keys())}")
        print(f"  图像: {tuple(sample['pixel_values'].shape)}")
        print(f"  文本: {tuple(sample['input_ids'].shape)}")
        print(f"  描述: {sample['caption'][:70]!r}")

        targets = list(FUSION_TYPES) if args.fusion == "all" else [normalize_fusion_type(args.fusion)]
        print(f"\n将要运行的融合方式: {targets}")

        results = {}
        for ft in targets:
            results[ft] = run_one_fusion(ft, ann, loaders, device, args)

        # ---------------------------------------------------------- 消融汇总
        if len(results) > 1:
            banner("消融实验汇总：四种融合方式对比")
            base = _load_single_modal_baselines()
            rows = []
            for name, met in base.items():
                rows.append((name, met))
            for ft, r in results.items():
                rows.append((f"融合:{ft}", r["test"]))

            print(f"{'模型':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'Top-5':>9}")
            print("-" * 70)
            for name, m in rows:
                print(f"{name:<22}{m.get('accuracy', float('nan')):>10.4f}"
                      f"{m.get('precision', float('nan')):>11.4f}"
                      f"{m.get('recall', float('nan')):>9.4f}"
                      f"{m.get('f1', float('nan')):>9.4f}"
                      f"{m.get('top5_accuracy', float('nan')):>9.4f}")

            best = max(results, key=lambda k: results[k]["test"]["accuracy"])
            print(f"\n最佳融合方式: {best}  "
                  f"acc={results[best]['test']['accuracy']:.4f}")

            summary = {
                "baselines": base,
                "fusions": {ft: r["test"] for ft, r in results.items()},
                "best_fusion": best,
                "config": {
                    "image_backbone": args.image_backbone,
                    "text_model": args.text_model,
                    "fusion_dim": args.fusion_dim,
                    "epochs": args.epochs,
                    "image_freeze": not args.no_freeze,
                    "text_freeze": not args.no_freeze_text,
                },
            }
            save_json(to_builtin(summary), config.RESULT_ROOT / "04_ablation_summary.json")
            print(f"汇总 -> {config.RESULT_ROOT / '04_ablation_summary.json'}")

            if HAS_MPL:
                viz.plot_ablation({k: v["test"] for k, v in results.items()},
                                  config.FIG_ROOT / "ablation_fusion.png",
                                  metric="accuracy", title="四种融合方式的测试集准确率")

            # 与单模态基线的对比图
            if base:
                allm = dict(base)
                allm.update({f"fusion:{k}": v["test"] for k, v in results.items()})
                viz.plot_ablation(allm, config.FIG_ROOT / "ablation_all_models.png",
                                  metric="accuracy", title="单模态 vs 多模态（测试集准确率）")

        section("完成")
        print("下一步: python 05_clip_zeroshot.py   （CLIP 思想实验：零样本分类与检索）")
        return 0
    finally:
        tee.close()


def _load_single_modal_baselines() -> dict:
    """读取单模态阶段的结果，用于和多模态融合同表对比。"""
    base = {}
    for pattern, name in ((("01_image_only_*.json"), "单模态:图像"),):
        for f in sorted(config.RESULT_ROOT.glob(pattern)):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                base[f"{name}({data.get('backbone', '?')})"] = data["test"]
            except Exception:
                pass
    for f in sorted(config.RESULT_ROOT.glob("02_text_only_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            base[f"单模态:文本({data.get('pooling', '?')})"] = data["test"]
        except Exception:
            pass
    return base


if __name__ == "__main__":
    sys.exit(main())
