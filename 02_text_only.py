"""02_text_only.py —— 阶段 2b 文本单模态训练（Text-Only）。

模型结构
--------
    BERT (bert-base-uncased)
      -> last_hidden_state [B, L, 768]
      -> attention_mask 过滤 padding 后做 mean pooling
      -> Linear(768, 100) 分类头

数据来源
--------
读 data/annotations.json 里的 splits（已按 6:2:2 切好）。
所有脚本读同一份划分，单模态与多模态的结果才可直接比较。

本脚本额外做的事
----------------
* 同时报告 [CLS] 池化与 mean 池化的对比，给出实测差异
* 输出每条描述的预测结果，用于错误分析（"描述太笼统时文本模型会错在哪"）
* 落盘训练曲线 / 混淆矩阵 / 错误分析

用法
----
    python 02_text_only.py                       # BERT + mean pooling
    python 02_text_only.py --pooling cls         # 换成 [CLS] 池化
    python 02_text_only.py --freeze              # 冻结 BERT，只训分类头（很快）
    python 02_text_only.py --epochs 2 --limit-batches 20   # 冒烟测试
    python 02_text_only.py --compare-pooling     # 训练后额外评估两种池化
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 先检查依赖：缺 torch 时给出可照做的提示，而不是一句 ModuleNotFoundError
from birdmm import require

require('torch', 'transformers', 'numpy', 'PIL')

import numpy as np
import torch
import torch.nn as nn

import config
from birdmm.data import Annotations, make_text_loaders
from birdmm.engine import evaluate, train_model
from birdmm.metrics import format_metrics, get_confusion_matrix, per_class_accuracy, top_confusions
from birdmm.text_encoder import TextClassifier, load_tokenizer
from birdmm.utils import (HAS_MPL, banner, count_parameters, human_num, log_to_file, save_json,
                          section, set_seed, to_builtin)
from birdmm import viz


def parse_args():
    p = argparse.ArgumentParser(description="文本单模态训练（BERT）")
    p.add_argument("--model-name", default=config.BERT_NAME, help="HuggingFace 模型名")
    p.add_argument("--pooling", default=config.TXT_POOLING, choices=["cls", "mean"],
                   help="句向量池化方式")
    p.add_argument("--epochs", type=int, default=config.TXT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.TXT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.TXT_LR)
    p.add_argument("--max-length", type=int, default=config.TEXT_MAX_LEN)
    p.add_argument("--freeze", action="store_true", help="冻结 BERT，只训练分类头")
    p.add_argument("--device", default=None)
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--compare-pooling", action="store_true",
                   help="训练后额外对比 cls / mean 两种池化的测试集表现")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tag = args.tag or f"bert_{args.pooling}" + ("_frozen" if args.freeze else "")
    tee = log_to_file(config.LOG_ROOT / f"02_text_only_{tag}.log")

    try:
        set_seed(config.SEED)
        device = config.get_device(args.device)

        banner("阶段 2b  文本单模态训练（Text-Only, BERT）")
        print(config.describe())
        print(f"模型={args.model_name}  pooling={args.pooling}  冻结BERT={args.freeze}  "
              f"lr={args.lr}  epochs={args.epochs}  batch={args.batch_size}  device={device}")

        # ---------------------------------------------------------- 数据
        section("1) 加载数据与 tokenizer")
        ann = Annotations()
        print(ann.summary())
        tokenizer = load_tokenizer(args.model_name)
        print(f"词表大小: {tokenizer.vocab_size}")

        # tokenizer 演示：句子 -> token -> id
        demo = "a small blue bird"
        enc = tokenizer(demo, return_tensors="pt")
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        print(f"\nTokenizer 演示: {demo!r}")
        print(f"  tokens   : {toks}")
        print(f"  input_ids: {enc['input_ids'][0].tolist()}")
        print(f"  特殊标记 : [CLS]={tokenizer.cls_token_id}  [SEP]={tokenizer.sep_token_id}  "
              f"[PAD]={tokenizer.pad_token_id}  [MASK]={tokenizer.mask_token_id}  "
              f"[UNK]={tokenizer.unk_token_id}")

        sets, loaders = make_text_loaders(ann, tokenizer, batch_size=args.batch_size)
        sample = sets["train"][0]
        print(f"\n单条文本张量: input_ids={tuple(sample['input_ids'].shape)}  "
              f"attention_mask={tuple(sample['attention_mask'].shape)}  "
              f"（max_length={args.max_length}）")
        real_tokens = int(sample["attention_mask"].sum().item())
        print(f"真实 token 数={real_tokens}，其余 {args.max_length - real_tokens} 个是 [PAD]"
              f" —— mean pooling 必须把它们屏蔽掉")

        # ---------------------------------------------------------- 模型
        section("2) 构建模型")
        model = TextClassifier(
            model_name=args.model_name,
            num_classes=ann.num_classes,
            pretrained=True,
            freeze=args.freeze,
            pooling=args.pooling,
            dropout=0.1,
        )
        total, trainable = count_parameters(model)
        print(model)
        print(f"总参数量     : {human_num(total)}")
        print(f"可训练参数量 : {human_num(trainable)}  ({trainable / total:.1%})")

        # ---------------------------------------------------------- 训练
        section("3) 训练")
        ckpt = config.CKPT_ROOT / f"02_text_{tag}.pt"
        result = train_model(
            model, loaders["train"], loaders["val"], device,
            epochs=args.epochs, lr=args.lr, num_classes=ann.num_classes,
            max_train_batches=args.limit_batches, max_eval_batches=args.limit_batches,
            tag=f"[txt:{args.pooling}]", save_path=ckpt,
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

        # ---------------------------------------------------------- 池化对比
        pooling_cmp = {}
        if args.compare_pooling:
            section("4b) 池化方式对比（取 [CLS] vs Mean Pooling）")
            for pool in ("cls", "mean"):
                m = TextClassifier(args.model_name, ann.num_classes, True, args.freeze,
                                   pooling=pool, dropout=0.1)
                # 只换池化、不换权重：把训练好的 BERT 权重搬过去
                state = {k: v for k, v in model.state_dict().items()
                         if not k.startswith("encoder.proj")}
                missing = m.load_state_dict(state, strict=False)
                m = m.to(device)
                met = evaluate(m, loaders["test"], criterion, device, num_classes=ann.num_classes)
                met.pop("loss", None)
                pooling_cmp[pool] = met
                print(f"  pooling={pool:<5} acc={met['accuracy']:.4f}  macro-F1={met['f1']:.4f}"
                      f"   (未匹配键: {len(missing.unexpected_keys)} 个)")
            best = max(pooling_cmp, key=lambda k: pooling_cmp[k]["accuracy"])
            print(f"  => 本数据集上 {best} 池化更好")

        # ---------------------------------------------------------- 保存
        section("5) 保存结果")
        out = {
            "phase": "stage-2b text-only",
            "model_name": args.model_name,
            "pooling": args.pooling,
            "freeze_bert": args.freeze,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "params_total": total,
            "params_trainable": trainable,
            "best_epoch": result["best_epoch"],
            "train_time_sec": round(result["train_time"], 1),
            "history": result["history"],
            "test": {k: v for k, v in test_metrics.items()},
            "pooling_comparison": pooling_cmp,
        }
        save_json(to_builtin(out), config.RESULT_ROOT / f"02_text_only_{tag}.json")
        print(f"指标 -> {config.RESULT_ROOT / f'02_text_only_{tag}.json'}")
        print(f"权重 -> {ckpt}")

        # ---------------------------------------------------------- 可视化
        if HAS_MPL:
            section("6) 可视化")
            viz.plot_train_curve(result["history"],
                                 config.FIG_ROOT / f"train_curve_text_{tag}.png",
                                 title=f"文本单模态 BERT({args.pooling})")
            cm = get_confusion_matrix(y_true, y_pred, ann.num_classes)
            viz.plot_confusion_matrix(cm, config.FIG_ROOT / f"confusion_matrix_text_{tag}.png",
                                      ann.classes, title=f"文本单模态 测试集 ({args.pooling})")
        else:
            print("\n[跳过可视化] 未安装 matplotlib")

        # ---------------------------------------------------------- 错误分析
        section("7) 错误分析")
        conf = top_confusions(y_true, y_pred, ann.classes, ann.num_classes, topn=10)
        print(f"{'真实类别':<34}{'被预测为':<34}{'次数':>6}")
        for c in conf:
            print(f"{c['true_class']:<34}{c['pred_class']:<34}{c['count']:>6}")

        accs = per_class_accuracy(y_true, y_pred, ann.num_classes)
        order = np.argsort(accs)
        print("\n准确率最低的 10 个类别（描述区分度最差的类）：")
        for i in order[:10]:
            print(f"  {ann.classes[i]:<34} acc={accs[i]:.3f}")

        # 展示几个具体错例 + 它们用的描述，方便分析"文本为什么不够"
        recs = ann.records("test")
        wrong = np.where(y_pred != y_true)[0]
        print(f"\n错例总数: {len(wrong)} / {len(y_true)}")
        if len(wrong):
            print("\n典型错例（含模型看到的描述）：")
            for idx in wrong[np.argsort(-probs[wrong].max(axis=1))][:5]:
                print(f"  描述   : {recs[idx]['captions'][0][:80]}")
                print(f"  真实   : {ann.classes[y_true[idx]]}")
                print(f"  预测   : {ann.classes[y_pred[idx]]}  (置信度 {probs[idx].max():.3f})")
                print()

        save_json(to_builtin({
            "top_confusions": conf,
            "lowest_accuracy_classes": [
                {"class": ann.classes[i], "accuracy": float(accs[i])} for i in order[:15]],
            "num_errors": int(len(wrong)),
        }), config.RESULT_ROOT / f"02_text_only_{tag}_errors.json")

        section("完成")
        print(f"测试集准确率: {test_metrics['accuracy']:.4f}   "
              f"宏平均 F1: {test_metrics['f1']:.4f}")
        print("下一步: python 03_model_dissection.py   （模型拆解与特征分析）")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    sys.exit(main())
