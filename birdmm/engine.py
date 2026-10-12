"""engine.py —— 通用训练/评估循环（三个单模态脚本 + 融合脚本共用）。

把"训练一个 epoch / 评估一个 split"的样板代码收在一处，
各脚本只负责：造数据 -> 造模型 -> 调 train_model() -> 报结果。
"""
from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .metrics import compute_metrics, format_metrics
from .utils import HAS_TQDM, format_time


# ---------------------------------------------------------------- 单轮训练
def train_one_epoch(model, loader, optimizer, criterion, device, epoch: int,
                    max_batches: Optional[int] = None, desc: str = "") -> Dict[str, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = 0

    iterator = loader
    if HAS_TQDM and not max_batches:
        from tqdm import tqdm

        iterator = tqdm(loader, desc=desc or f"Epoch {epoch}", leave=False)

    for step, batch in enumerate(iterator):
        if max_batches is not None and step >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        logits, labels = forward_batch(model, batch, device)
        loss = criterion(logits, labels)
        loss.backward()
        # 梯度裁剪：BERT 微调时防止偶发的梯度爆炸
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += bs
        n_batches += 1

    if total == 0:
        return {"loss": float("nan"), "accuracy": float("nan")}
    return {"loss": total_loss / total, "accuracy": correct / total}


# ---------------------------------------------------------------- 批量前向
def forward_batch(model, batch, device, task: str = "auto"):
    """统一把 batch 喂给模型，返回 (logits, labels)。

    task: 'image' / 'text' / 'multimodal' / 'auto'（按 batch 里有哪些键自动判断）
    """
    if task == "auto":
        if "pixel_values" in batch and "input_ids" in batch:
            task = "multimodal"
        elif "pixel_values" in batch:
            task = "image"
        else:
            task = "text"

    labels = batch["label"].to(device, non_blocking=True)

    if task == "image":
        logits = model(batch["pixel_values"].to(device, non_blocking=True))
    elif task == "text":
        logits = model(batch["input_ids"].to(device, non_blocking=True),
                       batch["attention_mask"].to(device, non_blocking=True))
    else:
        out = model(
            pixel_values=batch["pixel_values"].to(device, non_blocking=True),
            input_ids=batch["input_ids"].to(device, non_blocking=True),
            attention_mask=batch["attention_mask"].to(device, non_blocking=True),
        )
        logits = out["logits"] if isinstance(out, dict) else out

    return logits, labels


# ---------------------------------------------------------------- 评估
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int = 100,
             max_batches: Optional[int] = None, collect: bool = False) -> Dict:
    """在给定 split 上评估，返回指标字典。

    collect=True 时额外返回每条样本的预测/标签/概率，供错误分析与可视化使用
    （这些键以 `_` 开头，调用方通常 pop 掉后再展示）。
    """
    model.eval()
    total_loss, n = 0.0, 0
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        logits, labels = forward_batch(model, batch, device)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_logits.append(logits.detach().float().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    if not all_logits:
        return {"loss": float("nan")}

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = logits.argmax(axis=1)
    probs = _softmax(logits)

    metrics: Dict[str, Any] = dict(compute_metrics(y_true, y_pred, num_classes=num_classes,
                                                   logits=logits))
    metrics["loss"] = total_loss / max(n, 1)
    if collect:
        metrics["_y_true"] = y_true
        metrics["_y_pred"] = y_pred
        metrics["_probs"] = probs
        metrics["_logits"] = logits
    return metrics


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- 完整训练
def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    epochs: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_classes: int = 100,
    label_smoothing: float = 0.0,
    scheduler: str = "cosine",
    warmup_ratio: float = 0.05,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    tag: str = "",
    save_path=None,
    task: str = "auto",
) -> Dict:
    """标准训练流程：训练 -> 每轮验证 -> 保留最佳权重 -> 返回历史与最佳指标。

    Returns:
        {"history": [每轮 dict], "best_val": {...}, "best_epoch": int,
         "best_state": state_dict, "train_time": float}
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # 学习率调度按 **epoch** 推进（每个 epoch 结束后 step 一次），
    # 所以这里的单位是 epoch 而不是 step —— 写成 step 会让余弦退火几乎不动。
    warmup_epochs = max(int(round(epochs * warmup_ratio)), 1)
    total_epochs = max(epochs, 1)

    if scheduler == "cosine":
        def lr_lambda(epoch):
            # epoch 从 0 开始计（scheduler 在第一个 epoch 结束后才第一次 step）
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        sched = None

    history: List[Dict] = []
    best_val_acc, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # 记录本 epoch 实际使用的学习率（要在 sched.step() 之前取）
        epoch_lr = optimizer.param_groups[0]["lr"]

        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch,
                             max_batches=max_train_batches, desc=f"{tag} E{epoch}")
        if sched is not None:
            sched.step()

        val = evaluate(model, val_loader, criterion, device, num_classes=num_classes,
                       max_batches=max_eval_batches)
        rec = {"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["accuracy"],
               "val_loss": val.get("loss", float("nan")), "val_acc": val.get("accuracy", float("nan")),
               "val_f1": val.get("f1", float("nan")), "lr": epoch_lr}
        history.append(rec)

        star = ""
        if val.get("accuracy", 0) > best_val_acc:
            best_val_acc = val["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            star = "  ← best"
        print(f"  Epoch [{epoch:>2}/{epochs}]  train_loss={tr['loss']:.4f}  "
              f"train_acc={tr['accuracy']:.4f}  val_loss={rec['val_loss']:.4f}  "
              f"val_acc={rec['val_acc']:.4f}  val_f1={rec['val_f1']:.4f}{star}")

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path is not None:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, save_path)

    best_val = evaluate(model, val_loader, criterion, device, num_classes=num_classes,
                        max_batches=max_eval_batches)
    print(f"  [最佳] epoch={best_epoch}  val_acc={best_val.get('accuracy', float('nan')):.4f}"
          f"  训练耗时={format_time(train_time)}")

    return {"history": history, "best_val": best_val, "best_epoch": best_epoch,
            "best_state": best_state, "train_time": train_time, "model": model}
