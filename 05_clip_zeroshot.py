"""05_clip_zeroshot.py —— 阶段 5 CLIP 思想实验：从"分类器"到"共同语义空间"。

从"分类器"到"共同语义空间"的转变
-------------------
    阶段 4 的模型：  image + text ──► 融合 ──► 分类器 ──► 类别编号
    现代多模态模型：    image ──► 图像向量 ┐
                                          ├──► 余弦相似度最大者 = 匹配
                        text  ──► 文本向量 ┘

    **没有分类器了**。类别不是被"输出"的，而是通过"和类别描述的相似度"被比较出来的。

验证三种能力
-----------------------------
| 能力           | 做法                          | 本实验里怎么验证 |
| 图文匹配/检索  | image→text、text→image 排序   | 检索 R@1 / R@5   |
| 零样本分类     | 类别名写成提示词 → 比相似度   | 零样本 Accuracy  |
| 通用表示       | 换个任务不用重训              | 讨论题           |

本脚本全部实现，并且额外做三种"文本侧"的对照，回答"提示词到底值多少分"：
    A. 只写类别名          "black footed albatross"
    B. 手写模板提示词       "a photo of a black footed albatross"（+ 模板集成）
    C. 用真实描述做原型     该类别所有训练描述的向量平均（相当于"用语言描述当分类器"）

注意：全程 **不训练任何参数**，一次前向就算完 —— 这就是零样本的含义。

用法
----
    python 05_clip_zeroshot.py                     # 完整零样本 + 检索
    python 05_clip_zeroshot.py --limit-test 200    # 只测 200 张（快速看效果）
    python 05_clip_zeroshot.py --no-retrieval      # 跳过 R@K 检索
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

import config
from birdmm.data import Annotations, build_image_transform
from birdmm.utils import (HAS_MPL, Timer, banner, log_to_file, progress, save_json, section,
                          set_seed, to_builtin)
from birdmm import viz
from PIL import Image


# ------------------------------------------------------------------ 提示词模板
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a bird of the species {}.",
    "a picture of a {} in its natural habitat.",
    "a {} perched on a branch.",
    "this is a {}.",
    "a blurry photo of a {}.",
    "a good photo of a {}.",
    "a photo of a {} bird.",
]


def parse_args():
    p = argparse.ArgumentParser(description="CLIP 零样本分类与图文检索")
    p.add_argument("--clip-name", default=config.CLIP_NAME)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit-test", type=int, default=None, help="只评估前 N 张测试图")
    p.add_argument("--no-retrieval", action="store_true", help="跳过 R@K 检索评估")
    p.add_argument("--device", default=None)
    return p.parse_args()


# ------------------------------------------------------------------ CLIP 封装
def _to_tensor(out) -> torch.Tensor:
    """兼容不同 transformers 版本 get_*_features 的返回形式。

    transformers 5.x 返回 BaseModelOutputWithPooling（投影后的向量在 pooler_output），
    4.x 直接返回 Tensor。两种都支持，避免换个环境就崩。
    """
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "text_embeds", "image_embeds", "last_hidden_state"):
        v = getattr(out, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"无法从 {type(out)} 中取出特征张量")


class CLIPWrapper:
    """统一封装 CLIP 的图文编码（兼容 transformers 4.x / 5.x 的返回值差异）。"""

    def __init__(self, name: str, device):
        from transformers import CLIPModel, CLIPProcessor

        from birdmm.hub import load_pretrained

        self.device = device
        # 带离线回退：网络抖动时用本地缓存，不会让整个实验崩掉
        self.model = load_pretrained(CLIPModel.from_pretrained, name).to(device).eval()
        self.processor = load_pretrained(CLIPProcessor.from_pretrained, name)
        self.logit_scale = float(self.model.logit_scale.exp().detach().cpu())

    @property
    def embed_dim(self) -> int:
        return int(self.model.config.projection_dim)

    @property
    def image_mean(self):
        return tuple(self.processor.image_processor.image_mean)

    @property
    def image_std(self):
        return tuple(self.processor.image_processor.image_std)

    @torch.no_grad()
    def encode_images(self, images: list) -> np.ndarray:
        """images 必须是 **PIL 原图**：缩放与归一化交给 CLIPProcessor，
        否则会与它自带的归一化参数冲突（见 main 里的说明）。"""
        if isinstance(images[0], torch.Tensor):
            raise TypeError(
                "encode_images 需要 PIL 原图（CLIPProcessor 会自己做归一化）；"
                "收到 Tensor 说明调用方已经预处理过了，会导致归一化两次、特征失效。")
        inputs = self.processor(images=images, return_tensors="pt")
        pixel = inputs["pixel_values"].to(self.device)
        feats = _to_tensor(self.model.get_image_features(pixel_values=pixel))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    @torch.no_grad()
    def encode_texts(self, texts: list) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True,
                                max_length=77)
        ids = inputs["input_ids"].to(self.device)
        mask = inputs.get("attention_mask")
        mask = mask.to(self.device) if mask is not None else None
        feats = _to_tensor(self.model.get_text_features(input_ids=ids, attention_mask=mask))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()


def retrieval_at_k(sim: np.ndarray, k: int = 1) -> float:
    """sim: [N_query, N_gallery]，第 i 个 query 的正确 gallery 是第 i 个。"""
    n = sim.shape[0]
    k = min(k, sim.shape[1])
    topk = np.argsort(-sim, axis=1)[:, :k]
    return float((topk == np.arange(n)[:, None]).any(axis=1).mean())


def main() -> int:
    args = parse_args()
    tee = log_to_file(config.LOG_ROOT / "05_clip_zeroshot.log")

    try:
        set_seed(config.SEED)
        device = config.get_device(args.device)

        banner("阶段 5  CLIP 思想实验：零样本分类 + 图文检索")
        print(config.describe())
        print(f"CLIP 模型={args.clip_name}  device={device}")

        section("1) 加载 CLIP")
        with Timer("加载 CLIP"):
            clip = CLIPWrapper(args.clip_name, device)
        print(f"投影维度 (共同语义空间) : {clip.embed_dim}")
        print(f"可学习的温度 logit_scale: {clip.logit_scale:.2f}")
        print("注意：这里 **不训练任何参数**，所有结果都是一次前向算出来的 —— 这就是零样本。")

        ann = Annotations()
        print(f"\n{ann.summary()}")

        # ---------------------------------------------------------- 测试集
        section("2) 准备测试图像与类别描述")
        test_records = ann.records("test")
        if args.limit_test:
            test_records = test_records[:args.limit_test]

        # ⚠ 关键：CLIP 有自己的归一化参数，和 ImageNet 不一样
        #     CLIP    : mean=(0.4815, 0.4578, 0.4082)  std=(0.2686, 0.2613, 0.2758)
        #     ImageNet: mean=(0.4850, 0.4560, 0.4060)  std=(0.2290, 0.2240, 0.2250)
        #   如果把已经用 ImageNet 归一化过的 tensor 再交给 CLIPProcessor，
        #   就相当于归一化了两次，图像特征会完全失效（实测零样本 acc 从 0.72 掉到 0.03）。
        #   所以这里**保留 PIL 原图**，由 CLIPProcessor 自己做缩放+归一化。
        images, labels = [], []
        for rec in progress(test_records, desc="读取图像"):
            images.append(Image.open(config.DATA_ROOT / rec["image_path"]).convert("RGB"))
            labels.append(rec["label"])
        labels = np.array(labels)
        print(f"测试图像: {len(images)} 张")
        print(f"CLIP 官方归一化: mean={tuple(round(v, 4) for v in clip.image_mean)}  "
              f"std={tuple(round(v, 4) for v in clip.image_std)}")
        print(f"（与 ImageNet 的 mean={config.IMAGENET_MEAN} std={config.IMAGENET_STD} 不同，"
              f"所以原图交给 CLIPProcessor 处理，不要自己先归一化）")

        with Timer("图像编码"):
            image_feats = np.concatenate([
                clip.encode_images(images[i:i + args.batch_size])
                for i in range(0, len(images), args.batch_size)
            ], axis=0)
        print(f"图像特征: {image_feats.shape}  （每张图 -> {image_feats.shape[1]} 维单位向量）")

        # 三种文本侧的构造方式
        def cls_name(c: str) -> str:
            return c.split(".", 1)[1].replace("_", " ").lower() if "." in c else c.lower()

        text_a = [cls_name(c) for c in ann.classes]                       # 只写类别名
        text_b = [ann.class_prompts[c] for c in ann.classes]              # 单模板
        text_c = [f"a photo of a {cls_name(c)}." for c in ann.classes]

        variants = {
            "A. 只写类别名": text_a,
            "B. 单模板提示词": text_b,
            "C. 模板集成(10条)": None,          # 稍后特殊处理
        }

        section("3) 零样本分类：用类别描述当「分类器」")
        results = {}

        for name, texts in variants.items():
            if texts is None:
                continue
            tf = clip.encode_texts(texts)
            sim = image_feats @ tf.T                                       # [N, 100]
            pred = sim.argmax(axis=1)
            acc = float((pred == labels).mean())
            top5 = float(np.mean([
                labels[i] in np.argsort(-sim[i])[:5] for i in range(len(labels))]))
            results[name] = {"accuracy": acc, "top5_accuracy": top5,
                             "text_features": tf, "similarity": sim, "prompts": texts}
            print(f"  {name:<20} acc={acc:.4f}  top5={top5:.4f}")

        # C. 模板集成：把 10 个模板的文本向量平均（CLIP 论文里的 prompt ensembling）
        tmpl_feats = []
        for tmpl in PROMPT_TEMPLATES:
            texts = [tmpl.format(cls_name(c)) for c in ann.classes]
            tmpl_feats.append(clip.encode_texts(texts))
        ens = np.mean(tmpl_feats, axis=0)
        ens = ens / np.linalg.norm(ens, axis=1, keepdims=True)
        sim = image_feats @ ens.T
        pred = sim.argmax(axis=1)
        acc = float((pred == labels).mean())
        top5 = float(np.mean([labels[i] in np.argsort(-sim[i])[:5] for i in range(len(labels))]))
        results["C. 模板集成(10条)"] = {"accuracy": acc, "top5_accuracy": top5,
                                        "text_features": ens, "similarity": sim,
                                        "prompts": [t.format(cls_name(ann.classes[0]))
                                                    for t in PROMPT_TEMPLATES]}
        print(f"  {'C. 模板集成(10条)':<20} acc={acc:.4f}  top5={top5:.4f}")

        # D. 用真实描述做类别原型（把"语言描述"直接当分类器权重）
        section("4) 对照实验：用数据集里的真实描述当类别原型")
        proto_feats, proto_counts = [], []
        for label in range(ann.num_classes):
            caps = []
            for split in ("train", "val"):
                for rec in ann.splits[split]:
                    if rec["label"] == label:
                        caps.extend(rec["captions"])
            caps = caps[:200]                          # 每类最多取 200 条，控制耗时
            proto_counts.append(len(caps))
            f = np.concatenate([clip.encode_texts(caps[i:i + args.batch_size])
                                for i in range(0, len(caps), args.batch_size)], axis=0)
            f = f.mean(axis=0)
            proto_feats.append(f / max(np.linalg.norm(f), 1e-12))
        proto = np.stack(proto_feats)
        sim = image_feats @ proto.T
        pred = sim.argmax(axis=1)
        acc = float((pred == labels).mean())
        top5 = float(np.mean([labels[i] in np.argsort(-sim[i])[:5] for i in range(len(labels))]))
        results["D. 真实描述原型"] = {"accuracy": acc, "top5_accuracy": top5,
                                      "text_features": proto, "similarity": sim,
                                      "prompts": [f"<{ann.classes[i]} 的 {proto_counts[i]} 条描述平均>"
                                                  for i in range(ann.num_classes)]}
        print(f"  每类平均描述数: {np.mean(proto_counts):.0f}")
        print(f"  {'D. 真实描述原型':<20} acc={acc:.4f}  top5={top5:.4f}")
        print(f"  [说明] CUB 的描述是「这只鸟长什么样」的属性描述，几乎不提类别名，")
        print(f"         而 CLIP 的类别提示词直接含类别名 —— 所以 D 反而低于 B/C 很正常。")
        print(f"         这恰好说明：零样本分类效果高度依赖「文本侧怎么写」。")

        section("5) 零样本分类汇总（全程未训练任何参数）")
        print(f"{'文本侧构造方式':<24}{'Accuracy':>10}{'Top-5':>10}")
        print("-" * 46)
        for name, r in results.items():
            print(f"{name:<24}{r['accuracy']:>10.4f}{r['top5_accuracy']:>10.4f}")
        best = max(results, key=lambda k: results[k]["accuracy"])
        print(f"\n最佳: {best}  acc={results[best]['accuracy']:.4f}")
        print(f"（随机猜 100 类的期望准确率 = 0.01）")

        # ---------------------------------------------------------- 检索
        retrieval = {}
        # 这两个矩阵在检索段落里计算，但后面的可视化也要用，
        # 所以在外面先声明，避免"可能未绑定"的作用域问题。
        sim_i2t: np.ndarray | None = None
        text_feats: np.ndarray | None = None
        if not args.no_retrieval:
            section("6) 图文检索 R@K")
            # 用测试集的真实描述做 gallery
            captions = [rec["captions"][0] for rec in test_records]
            with Timer("文本编码"):
                text_feats = np.concatenate([
                    clip.encode_texts(captions[i:i + args.batch_size])
                    for i in range(0, len(captions), args.batch_size)], axis=0)

            sim_i2t = image_feats @ text_feats.T        # 图 -> 文
            sim_t2i = sim_i2t.T                          # 文 -> 图

            print(f"{'方向':<28}{'R@1':>8}{'R@5':>8}{'R@10':>8}")
            print("-" * 52)
            for direction, sim in (("图像 -> 文本 (image→text)", sim_i2t),
                                   ("文本 -> 图像 (text→image)", sim_t2i)):
                r1, r5, r10 = (retrieval_at_k(sim, 1), retrieval_at_k(sim, 5),
                               retrieval_at_k(sim, 10))
                retrieval[direction] = {"R@1": r1, "R@5": r5, "R@10": r10}
                print(f"{direction:<28}{r1:>8.4f}{r5:>8.4f}{r10:>8.4f}")
            print(f"\n（随机基线：R@1≈{1 / len(captions):.4f}，R@5≈{5 / len(captions):.4f}）")

            # 分类检索：用 100 个类别描述当 gallery，看正确答案能否排第一。
            # 注意这**不是** R@1 的意义所在：gallery 有 100 个候选，随机基线是 1/100，
            # 它和零样本分类的 accuracy 是同一个数（Top-1 命中率），这里只作为交叉验证。
            cls_sim = image_feats @ results["C. 模板集成(10条)"]["text_features"].T
            cls_top1 = float((cls_sim.argmax(axis=1) == labels).mean())
            print(f"\n交叉验证：用模板集成描述做 100 类检索的 Top-1 命中率 = {cls_top1:.4f}")
            print(f"           （与上表 C 的 accuracy 一致；100 类随机基线 = 0.0100）")
            retrieval["class_retrieval_top1"] = cls_top1

        # ---------------------------------------------------------- 可视化
        if HAS_MPL:
            section("7) 可视化")
            # 相似度矩阵：挑 12 张**不同类别**的图（测试集是按类别排序的，
            # 直接取前 12 张会全是同一类，矩阵就失去意义了）
            rng = np.random.default_rng(config.SEED)
            picked, seen_cls = [], set()
            for i in rng.permutation(len(labels)):
                if labels[i] not in seen_cls:
                    seen_cls.add(int(labels[i]))
                    picked.append(int(i))
                if len(picked) == 12:
                    break
            true_cls = [int(labels[i]) for i in picked]
            cand = list(dict.fromkeys(true_cls))[:6]
            others = [c for c in range(ann.num_classes) if c not in cand][:6]
            cand = cand + others
            cls_text_feats = results["C. 模板集成(10条)"]["text_features"]
            sub_sim = image_feats[picked] @ cls_text_feats[cand].T
            viz.plot_similarity_matrix(
                sub_sim, config.FIG_ROOT / "clip_similarity_matrix.png",
                row_labels=[ann.classes[true_cls[i]].split(".", 1)[1][:16]
                            for i in range(len(picked))],
                col_labels=[ann.classes[c].split(".", 1)[1][:16] for c in cand],
                title="CLIP 图文相似度矩阵（每行一张图，绿框=最高分；"
                      "前 6 列是该图的候选类，后 6 列是干扰类）")

            # 特征空间：图像特征按类着色
            viz.plot_embedding(image_feats, labels, config.FIG_ROOT / "clip_image_embedding.png",
                               class_names=ann.classes,
                               title="CLIP 图像特征空间（零样本可分的直观证据）", method="pca")

            # 对齐证据：配对 vs 非配对的相似度分布 + 正确描述的排名
            if sim_i2t is not None:
                diag = np.diag(sim_i2t)                       # 配对相似度
                off = sim_i2t[~np.eye(len(sim_i2t), dtype=bool)]
                # 每个图像的正确描述在 gallery 里的排名（1 = 排第一）
                ranks = (np.argsort(-sim_i2t, axis=1) == np.arange(len(sim_i2t))[:, None]
                         ).argmax(axis=1) + 1

                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

                axes[0].hist(off, bins=40, alpha=0.65, color="#C44E52",
                             label=f"非配对（随机图文）均值={off.mean():.3f}", density=True)
                axes[0].hist(diag, bins=25, alpha=0.75, color="#4C72B0",
                             label=f"配对（同一张图的图与文）均值={diag.mean():.3f}", density=True)
                axes[0].axvline(diag.mean(), color="#4C72B0", ls="--", lw=1.5)
                axes[0].axvline(off.mean(), color="#C44E52", ls="--", lw=1.5)
                axes[0].set_xlabel("cosine similarity"); axes[0].set_ylabel("密度")
                axes[0].set_title("配对的相似度明显更高 → 两个模态被对齐了", fontsize=10)
                axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

                n_gallery = len(sim_i2t)
                axes[1].hist(np.minimum(ranks, 21), bins=np.arange(1, 23) - 0.5,
                             color="#55A868", edgecolor="white")
                axes[1].set_xlabel("正确描述在候选里的排名（21 = 21 名及以后）")
                axes[1].set_ylabel("图像数量")
                axes[1].set_title(f"图文检索的排名分布（R@1={retrieval['图像 -> 文本 (image→text)']['R@1']:.3f}，"
                                  f"随机基线={1 / n_gallery:.4f}）", fontsize=10)
                axes[1].axvline(1.5, color="red", ls="--", lw=1, label="rank=1")
                axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

                fig.suptitle("CLIP 共同语义空间的对齐证据（对比阶段 3 未对齐时：配对与非配对没有差别）",
                             fontsize=11)
                fig.tight_layout(rect=(0, 0, 1, 0.92))
                viz._save(fig, config.FIG_ROOT / "clip_alignment_evidence.png")

                print(f"\n  配对相似度均值   = {diag.mean():+.4f}")
                print(f"  非配对相似度均值 = {off.mean():+.4f}")
                print(f"  差值             = {diag.mean() - off.mean():+.4f}"
                      f"   （阶段 3 未对齐时这个差值≈0）")
        else:
            print("\n[跳过可视化] 未安装 matplotlib")

        # ---------------------------------------------------------- 落盘
        section("8) 保存结果")
        out = {
            "phase": "stage-5 CLIP zero-shot",
            "clip_model": args.clip_name,
            "embed_dim": clip.embed_dim,
            "num_test_images": len(labels),
            "trained_parameters": 0,
            "zeroshot": {k: {"accuracy": v["accuracy"], "top5_accuracy": v["top5_accuracy"]}
                         for k, v in results.items()},
            "prompt_templates": PROMPT_TEMPLATES,
            "retrieval": retrieval,
        }
        save_json(to_builtin(out), config.RESULT_ROOT / "05_clip_zeroshot.json")
        print(f"结果 -> {config.RESULT_ROOT / '05_clip_zeroshot.json'}")

        # 保存特征，供报告/复现使用
        np.savez_compressed(config.FEATURE_ROOT / "clip_features.npz",
                            image_features=image_feats, labels=labels,
                            class_text_features=results["C. 模板集成(10条)"]["text_features"])
        print(f"特征 -> {config.FEATURE_ROOT / 'clip_features.npz'}")

        section("小结：为什么「没有分类器」也能分类")
        print(f"  1. CLIP 把图像和文本映射到同一个 {clip.embed_dim} 维语义空间（余弦相似度可比）")
        print(f"  2. 类别不再是分类头的第 k 个输出，而是「一段描述在这个空间里的位置」")
        print(f"  3. 因此换数据集不用重训：只要换类别描述即可（零样本）")
        print(f"  4. 文本侧写得越准，分数越高：")
        for name, r in results.items():
            print(f"       {name:<24} acc={r['accuracy']:.4f}")
        print(f"\n讨论题：")
        print(f"  * 为什么「模板集成」通常比「只写类别名」好？")
        print(f"  * 零样本分类和阶段 4 的融合分类，本质区别在哪？（提示：一个是「比较」，一个是「输出」）")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    sys.exit(main())
