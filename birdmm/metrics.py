"""metrics.py —— 分类性能指标。

优先用 scikit-learn（数值与教科书一致）；没装 sklearn 时自动退回纯 numpy 实现，
保证"可选依赖"这个承诺成立。

sklearn 的导入都放在函数内部：这样模块级没有条件导入，
静态检查不会报 "possibly unbound"，运行时也只在真正用到时才 import。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .utils import HAS_SKLEARN


# ------------------------------------------------------------ sklearn 调用封装
def _sk_prf(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> Dict[str, float]:
    """用 sklearn 算 accuracy / precision / recall / f1。

    zero_division 在 sklearn 里的类型标注是 str，但实际接受 0 / 1 / "warn"，
    所以这里显式标注类型绕过 stub 的过度约束。
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    zd: Any = 0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=zd)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=zd)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=zd)),
    }


def _sk_topk(y_true: np.ndarray, logits: np.ndarray, k: int) -> float:
    from sklearn.metrics import top_k_accuracy_score

    return float(top_k_accuracy_score(y_true, logits, k=k,
                                      labels=np.arange(logits.shape[1])))


def _sk_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))


def _as_array(x) -> np.ndarray:
    """统一转成 numpy 数组（兼容 torch tensor / list / ndarray）。"""
    if hasattr(x, "detach"):        # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# ------------------------------------------------------------ numpy 兜底实现
def _np_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _np_prf(cm: np.ndarray, average: str = "macro") -> Dict[str, float]:
    """从混淆矩阵算 precision / recall / f1（zero_division=0）。"""
    tp = np.diag(cm).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)     # 每列 = 被预测为该类的总数
    true_sum = cm.sum(axis=1).astype(np.float64)     # 每行 = 该类真实样本数

    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    recall = np.divide(tp, true_sum, out=np.zeros_like(tp), where=true_sum > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    if average == "macro":
        return {"precision": float(precision.mean()),
                "recall": float(recall.mean()),
                "f1": float(f1.mean())}
    if average == "weighted":
        w = true_sum / max(true_sum.sum(), 1)
        return {"precision": float((precision * w).sum()),
                "recall": float((recall * w).sum()),
                "f1": float((f1 * w).sum())}
    if average == "micro":
        acc = float(tp.sum() / max(cm.sum(), 1))
        return {"precision": acc, "recall": acc, "f1": acc}
    raise ValueError(f"不支持的 average: {average}")


# ------------------------------------------------------------------ 对外接口
def compute_metrics(
    y_true,
    y_pred,
    num_classes: int = 100,
    logits: np.ndarray | None = None,
    average: str = "macro",
) -> Dict[str, float]:
    """计算全套指标。

    Args:
        y_true:  真实标签 [N]
        y_pred:  预测标签 [N]
        logits:  [N, C] 的模型输出，给了才能算 Top-5
        average: macro / weighted / micro
    """
    yt = _as_array(y_true).astype(np.int64).ravel()
    yp = _as_array(y_pred).astype(np.int64).ravel()

    out: Dict[str, float] = {}

    if HAS_SKLEARN:
        out.update(_sk_prf(yt, yp, average))
    else:
        cm = _np_confusion_matrix(yt, yp, num_classes)
        out["accuracy"] = float(np.diag(cm).sum() / max(cm.sum(), 1))
        out.update(_np_prf(cm, average))

    if logits is not None:
        lg = _as_array(logits)
        k = min(5, lg.shape[1])
        if HAS_SKLEARN:
            try:
                out[f"top{k}_accuracy"] = _sk_topk(yt, lg, k)
            except Exception:
                out[f"top{k}_accuracy"] = _np_topk(yt, lg, k)
        else:
            out[f"top{k}_accuracy"] = _np_topk(yt, lg, k)

    out["num_samples"] = int(len(yt))
    return out


def _np_topk(y_true: np.ndarray, logits: np.ndarray, k: int) -> float:
    topk = np.argsort(-logits, axis=1)[:, :k]
    hit = (topk == y_true[:, None]).any(axis=1)
    return float(hit.mean())


def get_confusion_matrix(y_true, y_pred, num_classes: int = 100) -> np.ndarray:
    yt = _as_array(y_true).astype(np.int64).ravel()
    yp = _as_array(y_pred).astype(np.int64).ravel()
    if HAS_SKLEARN:
        return _sk_confusion(yt, yp, num_classes)
    return _np_confusion_matrix(yt, yp, num_classes)


def per_class_accuracy(y_true, y_pred, num_classes: int = 100) -> np.ndarray:
    """每一类的准确率，用于找出"最容易被混淆的类别"。"""
    cm = get_confusion_matrix(y_true, y_pred, num_classes)
    true_sum = cm.sum(axis=1)
    return np.divide(np.diag(cm), true_sum, out=np.zeros(num_classes, dtype=np.float64),
                     where=true_sum > 0)


def top_confusions(y_true, y_pred, class_names: List[str], num_classes: int = 100, topn: int = 10):
    """返回最常被搞混的 (真实类 -> 预测类) 组合，供错误分析使用。"""
    cm = get_confusion_matrix(y_true, y_pred, num_classes)
    np.fill_diagonal(cm, 0)
    flat = np.argsort(-cm, axis=None)[:topn]
    out = []
    for idx in flat:
        i, j = divmod(int(idx), num_classes)
        if cm[i, j] == 0:
            break
        out.append({
            "true_label": i,
            "true_class": class_names[i] if i < len(class_names) else str(i),
            "pred_label": j,
            "pred_class": class_names[j] if j < len(class_names) else str(j),
            "count": int(cm[i, j]),
        })
    return out


def format_metrics(m: Dict[str, float]) -> str:
    """把指标字典格式化成一个易读的表格。"""
    order = ["accuracy", "precision", "recall", "f1", "top5_accuracy", "num_samples"]
    label = {
        "accuracy": "准确率 (Accuracy)",
        "precision": "宏平均精确率 (Precision)",
        "recall": "宏平均召回率 (Recall)",
        "f1": "宏平均 F1 分数 (F1-Score)",
        "top5_accuracy": "Top-5 准确率",
        "num_samples": "样本数",
    }
    lines = ["=" * 44]
    for k in order:
        if k not in m:
            continue
        v = m[k]
        if k == "num_samples":
            lines.append(f"{label[k]:<28}: {int(v)}")
        else:
            lines.append(f"{label[k]:<28}: {v:.4f}")
    lines.append("=" * 44)
    return "\n".join(lines)
