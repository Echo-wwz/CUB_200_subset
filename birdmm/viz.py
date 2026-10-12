"""viz.py —— 可视化工具（全部基于 matplotlib，未安装时自动跳过）。

生成的图（outputs/figures/）：
    train_curve_*.png          训练/验证曲线
    confusion_matrix_*.png     混淆矩阵（行归一化）
    feature_map_*.png          ResNet 空间特征图（模型拆解）
    attention_*.png            跨模态注意力热力图（模型在看哪个词）
    embedding_pca_*.png        图文特征空间的 PCA / t-SNE 投影（CLIP 思想实验）
    gate_distribution_*.png    门控融合的 gate 分布
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .utils import HAS_MPL

if HAS_MPL:
    import matplotlib

    matplotlib.use("Agg")               # 无界面后端：在 VS Code / 服务器上都不会弹窗卡住
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 让中文标题正常显示（找不到中文字体就退回英文，不会报错）
    _CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
    _available = {f.name for f in font_manager.fontManager.ttflist}
    for _f in _CJK_CANDIDATES:
        if _f in _available:
            plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


def _require_mpl() -> bool:
    if not HAS_MPL:
        print("[跳过绘图] 未安装 matplotlib（pip install matplotlib 后可用）")
        return False
    return True


def _save(fig, path: Path) -> Optional[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if HAS_MPL:
        import matplotlib.pyplot as plt

        plt.close(fig)
    print(f"[图] {path}")
    return path


# ------------------------------------------------------------------ 训练曲线
def plot_train_curve(history: Sequence[Dict], path: Path, title: str = "训练曲线"):
    if not _require_mpl():
        return None
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="train loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "s-", label="val loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title} - Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["train_acc"] for h in history], "o-", label="train acc")
    axes[1].plot(epochs, [h["val_acc"] for h in history], "s-", label="val acc")
    if "val_f1" in history[0]:
        axes[1].plot(epochs, [h["val_f1"] for h in history], "^--", label="val macro-F1")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"{title} - Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    return _save(fig, path)


# ------------------------------------------------------------------ 混淆矩阵
def plot_confusion_matrix(cm: np.ndarray, path: Path, class_names: Optional[List[str]] = None,
                          title: str = "混淆矩阵", normalize: bool = True, max_classes: int = 100):
    if not _require_mpl():
        return None
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)

    n = cm.shape[0]
    show_labels = n <= 30
    fig, ax = plt.subplots(figsize=(9, 8) if not show_labels else (10, 9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title(f"{title}（{n} 类{'，行归一化' if normalize else ''}）")
    if show_labels and class_names:
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
    else:
        step = max(n // 10, 1)
        ticks = list(range(0, n, step))
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([str(t + 1) for t in ticks], fontsize=8)
        ax.set_yticklabels([str(t + 1) for t in ticks], fontsize=8)
    return _save(fig, path)


# ------------------------------------------------------------------ 特征图
def plot_feature_maps(fmap: np.ndarray, path: Path, title: str = "ResNet 空间特征图",
                      max_channels: int = 16, original: Optional[np.ndarray] = None):
    """fmap: [C, h, w]；original: [3, H, W]（可选的原始图像，已反标准化）。"""
    if not _require_mpl():
        return None
    import matplotlib.pyplot as plt

    fmap = np.asarray(fmap)
    if fmap.ndim == 4:
        fmap = fmap[0]
    c, h, w = fmap.shape
    k = min(max_channels, c)

    ncols = 6
    nrows = int(np.ceil((k + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    ax0 = axes[0]
    if original is not None:
        img = np.asarray(original).transpose(1, 2, 0) if original.ndim == 3 else original
        img = (img - img.min()) / max(img.max() - img.min(), 1e-6)
        ax0.imshow(img)
    else:
        ax0.imshow(fmap.mean(axis=0), cmap="viridis")
    ax0.set_title("原图" if original is not None else "通道均值", fontsize=9)

    # 按"激活强度"挑最有代表性的通道
    energy = fmap.reshape(c, -1).std(axis=1)
    top = np.argsort(-energy)[:k]
    for i, ch in enumerate(top):
        ax = axes[i + 1]
        ax.imshow(fmap[ch], cmap="viridis")
        ax.set_title(f"ch{ch}", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{title}  [C={c}, {h}x{w}]  通道方差 Top-{k}", fontsize=11)
    return _save(fig, path)


# ------------------------------------------------------------------ 注意力
def plot_attention(attn: np.ndarray, tokens: Sequence[str], path: Path,
                   title: str = "跨模态注意力（图像位置 → 文本词）", grid: tuple = (7, 7),
                   head: int = 0):
    """attn: [heads, 49, L] 或 [49, L]；tokens: 长度 L 的字符串列表。

    左图：每个文本 token 从 49 个图像位置收到的平均注意力（模型在看哪些词）
    右图：7x7 网格 —— 每个图像位置最关注的词，颜色 = 该位置的注意力集中程度
          （熵越低说明这个位置越"专一"地盯着某一个词）
    """
    if not _require_mpl():
        return None
    import matplotlib.pyplot as plt

    attn = np.asarray(attn)
    if attn.ndim == 3:
        attn = attn[head]
    n_tok = min(len(tokens), attn.shape[1])
    attn = attn[:, :n_tok]
    tokens = list(tokens)[:n_tok]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # ---- 左：每个文本 token 的平均注意力
    per_token = attn.mean(axis=0)
    order = np.argsort(-per_token)
    colors = ["#C44E52" if tokens[i] in ("[CLS]", "[SEP]", "[PAD]") else "#4C72B0"
              for i in order]
    axes[0].barh(range(n_tok), per_token[order], color=colors)
    axes[0].set_yticks(range(n_tok))
    axes[0].set_yticklabels([tokens[i] for i in order], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("平均注意力权重")
    axes[0].set_title("模型在「看」哪些词（红=特殊标记）", fontsize=10)
    axes[0].grid(axis="x", alpha=0.3)

    # ---- 右：7x7 网格，每个位置最关注的词 + 该位置注意力的集中度
    best = attn.argmax(axis=1)
    p = attn / np.clip(attn.sum(axis=1, keepdims=True), 1e-12, None)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)              # 每个位置的注意力熵
    ent_norm = ent / max(np.log(n_tok), 1e-12)              # 归一到 [0,1]

    h, w = grid
    if h * w == attn.shape[0]:
        gridmap = best.reshape(h, w)
        entmap = ent_norm.reshape(h, w)
        im = axes[1].imshow(entmap, cmap="RdYlGn_r", vmin=0, vmax=1)
        n_distinct = len(np.unique(best))
        for i in range(h):
            for j in range(w):
                t = tokens[gridmap[i, j]]
                axes[1].text(j, i, t[:7], ha="center", va="center", fontsize=5.5,
                             color="black",
                             bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none",
                                       alpha=0.65))
        note = "" if n_distinct > 1 else f"（所有位置都指向 {tokens[best[0]]}）"
        axes[1].set_title(f"每个图像位置最关注的词（7×7）{note}\n"
                          f"底色 = 注意力集中度（越绿越专一，越红越分散）", fontsize=10)
        axes[1].set_xticks([]); axes[1].set_yticks([])
        fig.colorbar(im, ax=axes[1], fraction=0.046, label="归一化注意力熵")
    else:
        im = axes[1].imshow(attn, aspect="auto", cmap="viridis")
        axes[1].set_xlabel("文本 token"); axes[1].set_ylabel("图像位置")
        axes[1].set_xticks(range(n_tok)); axes[1].set_xticklabels(tokens, rotation=90, fontsize=7)
        fig.colorbar(im, ax=axes[1], fraction=0.046)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))       # 给 suptitle 留位置，避免和子图标题重叠
    return _save(fig, path)


# ------------------------------------------------------------------ 特征空间
def plot_embedding(features: np.ndarray, labels: np.ndarray, path: Path,
                   class_names: Optional[List[str]] = None, title: str = "特征空间投影",
                   method: str = "pca", max_points: int = 2000, seed: int = 42):
    """把高维特征降到 2D 看"同类是否聚在一起"（CLIP 共同语义空间的直观验证）。"""
    if not _require_mpl():
        return None
    import matplotlib.pyplot as plt

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if len(features) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(features), max_points, replace=False)
        features, labels = features[idx], labels[idx]

    emb = None
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            emb = TSNE(n_components=2, perplexity=min(30, len(features) - 1),
                       init="pca", random_state=seed).fit_transform(features)
        except Exception as exc:
            print(f"[提示] t-SNE 不可用（{exc}），改用 PCA")
            method = "pca"
    if method == "pca":
        x = features - features.mean(axis=0, keepdims=True)
        # SVD 版 PCA：不需要 sklearn
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        emb = u[:, :2] * s[:2]
        explained = (s[:2] ** 2) / max((s ** 2).sum(), 1e-12)
        method = f"pca (解释方差 {explained[0]:.1%} + {explained[1]:.1%})"

    if emb is None:                      # 兜底：method 传了别的值
        print(f"[跳过绘图] 未知的降维方法 '{method}'")
        return None

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab20")
    uniq = np.unique(labels)
    for i, lab in enumerate(uniq):
        m = labels == lab
        name = class_names[lab] if class_names and lab < len(class_names) else str(lab)
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.7, color=cmap(i % 20),
                   label=name if len(uniq) <= 20 else None)
    if len(uniq) <= 20:
        ax.legend(fontsize=7, markerscale=2, ncol=2)
    ax.set_title(f"{title}（{method}，{len(uniq)} 类）")
    ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2"); ax.grid(alpha=0.2)
    return _save(fig, path)


# ------------------------------------------------------------------ 相似度矩阵
def plot_similarity_matrix(sim: np.ndarray, path: Path,
                           row_labels: Sequence[str] | None = None,
                           col_labels: Sequence[str] | None = None,
                           title: str = "图文相似度矩阵"):
    """CLIP 零样本的核心图：每张图对每个类别提示词的余弦相似度。"""
    if not _require_mpl():
        return None
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(max(6, 0.42 * sim.shape[1] + 3),
                                    max(4, 0.34 * sim.shape[0] + 2)))
    im = ax.imshow(sim, cmap="RdYlBu_r", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, label="cosine similarity")
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels([str(s)[:28] for s in row_labels], fontsize=7)
    if col_labels is not None:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels([str(s)[:24] for s in col_labels], rotation=90, fontsize=7)
    for i in range(sim.shape[0]):
        j = int(np.argmax(sim[i]))
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="lime", lw=2))
    ax.set_title(title)
    return _save(fig, path)


# ------------------------------------------------------------------ 门控分布
def plot_gate_distribution(gates: np.ndarray, labels: np.ndarray, path: Path,
                           class_names: Optional[List[str]] = None,
                           title: str = "门控融合的 gate 分布"):
    """gates: [N] 或 [N, d]（每样本的 gate 均值）；labels: [N]。"""
    if not _require_mpl():
        return None
    import matplotlib.pyplot as plt

    gates = np.asarray(gates)
    if gates.ndim > 1:
        gates = gates.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].hist(gates, bins=40, color="#55A868", edgecolor="white")
    axes[0].axvline(0.5, color="red", ls="--", lw=1, label="0.5（两者等权）")
    axes[0].set_xlabel("gate 值 a（→1 更信图像，→0 更信文本）")
    axes[0].set_ylabel("样本数"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"gate 均值={gates.mean():.3f}（>0.5 表示整体更依赖图像）")

    # 每类的平均 gate：看哪些类别更依赖图像
    uniq = np.unique(labels)
    means = np.array([gates[labels == lab].mean() for lab in uniq])
    order = np.argsort(means)

    # 取两端各 12 个类别。若所有类的 gate 都在 0.5 同一侧，
    # 就按"相对大小"染色，否则"最依赖文本"那一栏会全是蓝的、失去区分度。
    k = min(12, len(order) // 2 if len(order) >= 4 else len(order))
    show = (order[:k].tolist() + order[-k:].tolist()) if len(order) >= 4 else order.tolist()
    show = sorted(set(show), key=lambda i: means[i])

    both_sides = means.min() < 0.5 < means.max()
    if both_sides:
        colors = ["#C44E52" if means[i] < 0.5 else "#4C72B0" for i in show]
        subtitle = "最依赖文本(红) / 最依赖图像(蓝) 的类别"
    else:
        # 全部偏向同一侧：用深浅表示相对倾向
        lo, hi = means[show].min(), means[show].max()
        span = max(hi - lo, 1e-6)
        colors = [plt.get_cmap("coolwarm_r")((means[i] - lo) / span) for i in show]
        side = "图像" if means.mean() > 0.5 else "文本"
        subtitle = f"各类别对图像的依赖程度（整体偏{side}，越深越依赖图像）"

    names = [(class_names[uniq[i]] if class_names and uniq[i] < len(class_names) else str(uniq[i]))
             for i in show]
    axes[1].barh(range(len(show)), means[show], color=colors)
    axes[1].set_yticks(range(len(show)))
    axes[1].set_yticklabels([n[:24] for n in names], fontsize=8)
    axes[1].axvline(0.5, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("该类平均 gate"); axes[1].grid(axis="x", alpha=0.3)
    axes[1].set_title(subtitle)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


# ------------------------------------------------------------------ 消融柱状图
def plot_ablation(results: Dict[str, Dict[str, float]], path: Path, metric: str = "accuracy",
                  title: str = "融合方式对比"):
    if not _require_mpl():
        return None
    import matplotlib.pyplot as plt

    names = list(results.keys())
    vals = [results[n].get(metric, float("nan")) for n in names]
    order = np.argsort(vals)[::-1]
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4.8))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    bars = ax.bar(range(len(names)), vals, color=[colors[i % len(colors)] for i in range(len(names))])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names)))
    # 标签较长时旋转，避免互相重叠
    longest = max((len(n) for n in names), default=0)
    rotation = 0 if longest <= 10 and len(names) <= 6 else 20
    ax.set_xticklabels(names, fontsize=9.5, rotation=rotation,
                       ha="right" if rotation else "center")
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
    ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------------ 样本网格
def plot_sample_grid(images: Sequence[np.ndarray], titles: Sequence[str], path: Path,
                     ncols: int = 5, title: str = ""):
    """images: 一组 [3,H,W] 的数组（已反标准化或原图）。"""
    if not _require_mpl():
        return None
    import matplotlib.pyplot as plt

    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, (img, t) in enumerate(zip(images, titles)):
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
        axes[i].imshow(arr, cmap=None if arr.ndim == 3 else "gray")
        axes[i].set_title(t, fontsize=7.5)
        axes[i].axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    return _save(fig, path)
