"""03_model_dissection.py —— 阶段 3 模型拆解：把两个编码器拆开看特征。

把图像编码器与文本编码器分别拆开，看它们内部到底做了什么。

脚本做的事
----------
【图像侧】
  1. 逐层打印 ResNet 的形状变化：图片 [3,224,224] -> 特征图 [512,7,7]
  2. 推导特征图到 token 序列的转换：
         [B,512,7,7] -> flatten(2) -> [B,512,49] -> transpose(1,2) -> [B,49,512]
     并**数值验证**两种写法完全等价（flatten+transpose 与 reshape+permute）
  3. 全局平均池化 [B,512] vs 展平成 49 个 token [B,49,512] —— 两种用法各自的形状与用途
  4. 可视化空间特征图（哪些通道在看鸟的哪个部位）

【文本侧】
  5. Tokenizer 演示：句子 -> token -> id（含 [CLS]/[SEP]/[PAD]）
  6. 三层 embedding 相加：Token + Segment + Position（打印各自的形状与范数）
  7. BERT 输出 [B,L,768]，两种池化：[CLS] vs masked mean
  8. **数值演示 mask 的必要性**：不做 mask 的 mean pooling 被 [PAD] 稀释了多少

【跨模态】
  9. 图像向量 [B,512] 与文本向量 [B,768]：维度不同、空间不同
 10. 直接算余弦相似度 -> 证明"没有对齐训练的两个空间根本不可比"（为阶段 5 的 CLIP 做铺垫）
 11. 把两个模态的特征各自降维投影，看它们的分布完全分离

用法
----
    python 03_model_dissection.py
    python 03_model_dissection.py --num-samples 8 --image-ckpt outputs/checkpoints/01_image_resnet18.pt
    python 03_model_dissection.py --text "a small blue bird with a long beak"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 先检查依赖：缺 torch 时给出可照做的提示，而不是一句 ModuleNotFoundError
from birdmm import require

require('torch', 'torchvision', 'transformers', 'numpy', 'PIL')

import numpy as np
import torch
import torch.nn as nn

from PIL import Image

import config
from birdmm.data import Annotations, build_image_transform
from birdmm.image_encoder import ImageEncoder
from birdmm.text_encoder import TextEncoder, load_tokenizer, masked_mean
from birdmm.utils import (HAS_MPL, banner, log_to_file, save_json, section, set_seed, to_builtin)
from birdmm import viz


def parse_args():
    p = argparse.ArgumentParser(description="模型拆解：图像/文本特征分析")
    p.add_argument("--image-ckpt", default=None, help="01_image_only.py 训练出的权重（可选）")
    p.add_argument("--backbone", default=config.IMG_BACKBONE)
    p.add_argument("--text-model", default=config.BERT_NAME)
    p.add_argument("--num-samples", type=int, default=6, help="分析多少张图")
    p.add_argument("--text", default="a small blue bird with a long beak",
                   help="用于 tokenizer / BERT 演示的句子")
    p.add_argument("--device", default=None)
    return p.parse_args()


def describe_module_shapes(encoder: ImageEncoder, device) -> list[dict]:
    """逐层前向，记录每一步输出的形状 —— 把 ResNet 拆开看。"""
    x = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    records = []

    def hook(name):
        def fn(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                records.append({"stage": name, "shape": list(out.shape)})
        return fn

    handles = []
    names = ["stem(conv1+bn+relu+maxpool)"] + [f"layer{i}" for i in range(1, 5)]
    children = list(encoder.body.children())
    for name, child in zip(names, children):
        handles.append(child.register_forward_hook(hook(name)))

    with torch.no_grad():
        fmap = encoder.body(x)

    for h in handles:
        h.remove()
    records.append({"stage": "最终特征图 body(x)", "shape": list(fmap.shape)})
    return records


def main() -> int:
    args = parse_args()
    tee = log_to_file(config.LOG_ROOT / "03_model_dissection.log")

    try:
        set_seed(config.SEED)
        device = config.get_device(args.device)

        banner("阶段 3  模型拆解：图像编码器与文本编码器")
        print(config.describe())
        print(f"device={device}")

        ann = Annotations()
        stats: dict = {}

        # ============================================================ 图像侧
        section("【图像侧】1) Image Encoder 逐层形状")
        encoder = ImageEncoder(args.backbone, pretrained=True, freeze=True).to(device)
        encoder.eval()

        if args.image_ckpt and Path(args.image_ckpt).exists():
            state = torch.load(args.image_ckpt, map_location="cpu")
            own = encoder.state_dict()
            matched = {k: v for k, v in state.items()
                       if k.startswith("encoder.") and k[len("encoder."):] in own}
            encoder.load_state_dict({k[len("encoder."):]: v for k, v in matched.items()},
                                    strict=False)
            print(f"已加载图像权重: {args.image_ckpt}（匹配 {len(matched)} 个张量）")

        print(f"\n输入: 图片 [B, 3, {config.IMAGE_SIZE}, {config.IMAGE_SIZE}]")
        shapes = describe_module_shapes(encoder, device)
        for r in shapes:
            print(f"    {r['stage']:<28} -> {r['shape']}")
        stats["image_layer_shapes"] = shapes

        section("2) 推导验证：特征图 -> token 序列")
        x = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
        with torch.no_grad():
            fmap = encoder.body(x)                                  # [B, C, h, w]
            B, C, h, w = fmap.shape
            print(f"    特征图                : {list(fmap.shape)}   (B, C, h, w)")
            t1 = fmap.flatten(2).transpose(1, 2)                    # 写法一
            print(f"    flatten(2)            : {list(fmap.flatten(2).shape)}   (B, C, h*w)")
            print(f"    .transpose(1, 2)      : {list(t1.shape)}   (B, h*w, C)")
            t2 = fmap.reshape(B, C, -1).permute(0, 2, 1)            # 等价写法
            print(f"    reshape+permute       : {list(t2.shape)}   (B, h*w, C)")
            same = torch.allclose(t1, t2)
            print(f"    两种写法数值完全一致  : {same}")
            print(f"    => 一张 224x224 的图被切成 {h * w} 个「视觉单词」，每个 {C} 维")
            stats["token_equivalence"] = bool(same)

        section("3) 两种用法：全局向量 vs token 序列")
        with torch.no_grad():
            pooled = encoder(x)                                     # [B, out_dim]
            tokens = encoder(x, return_tokens=True)                 # [B, h*w, out_dim]
        print(f"    全局平均池化 (AvgPooling) : {list(pooled.shape)}  "
              f"-> Early / Late / Gated 融合用")
        print(f"    展平成 token 序列         : {list(tokens.shape)}  "
              f"-> Cross-Attention 融合用（Query）")
        print(f"    token 数与特征图位置数一致: {tokens.shape[1] == h * w}")
        stats["image_feature_dim"] = int(pooled.shape[1])
        stats["num_visual_tokens"] = int(tokens.shape[1])

        # ============================================================ 真实图像
        section("4) 在真实图片上提取特征并可视化")
        transform = build_image_transform(train=False)
        records = ann.records("test")[:args.num_samples]
        images, titles = [], []
        feats = []
        for rec in records:
            img = Image.open(config.DATA_ROOT / rec["image_path"]).convert("RGB")
            # torchvision 的 transform 返回类型在 stub 里是 Image | Tensor，
            # 静态检查推断不出来，这里显式 cast 成 Tensor（运行时确实是 Tensor）
            tensor = cast(torch.Tensor, transform(img)).unsqueeze(0).to(device)
            with torch.no_grad():
                f, fm = encoder.forward_maps(tensor)
            feats.append(f.squeeze(0).cpu().numpy())
            images.append(tensor.squeeze(0).cpu().numpy())
            titles.append(f"{rec['class_name'][:18]}\n|f|={f.norm().item():.1f}")

        feats = np.stack(feats)
        print(f"    提取了 {len(feats)} 张图的特征，形状 {feats.shape}")
        print(f"    特征范数: 均值={np.linalg.norm(feats, axis=1).mean():.3f}  "
              f"最小={np.linalg.norm(feats, axis=1).min():.3f}  "
              f"最大={np.linalg.norm(feats, axis=1).max():.3f}")
        print(f"    各维度标准差均值: {feats.std(axis=0).mean():.4f}（越小说明该维度越不活跃）")

        if HAS_MPL:
            # 取第一张图做特征图可视化
            tensor = torch.tensor(images[0]).unsqueeze(0).to(device)
            with torch.no_grad():
                _, fm = encoder.forward_maps(tensor)
            fm_np = fm.squeeze(0).cpu().numpy()                     # [512,7,7]
            viz.plot_feature_maps(fm_np, config.FIG_ROOT / "feature_map_resnet.png",
                                  title=f"{args.backbone} 特征图（{records[0]['class_name']}）",
                                  original=images[0])
            viz.plot_sample_grid(images, titles, config.FIG_ROOT / "dissection_samples.png",
                                 ncols=3, title="参与特征分析的样本")

        # 保存特征供后续脚本复用（例如做检索 / 融合输入）
        np.savez_compressed(config.FEATURE_ROOT / "image_features.npz",
                            features=feats, labels=np.array([r["label"] for r in records]),
                            paths=np.array([r["image_path"] for r in records]))
        print(f"    特征已存 -> {config.FEATURE_ROOT / 'image_features.npz'}")

        # ============================================================ 文本侧
        section("5) Text Encoder：Tokenizer 演示")
        tokenizer = load_tokenizer(args.text_model)
        sentence = args.text
        enc = tokenizer(sentence, return_tensors="pt")
        ids = enc["input_ids"][0]
        toks = tokenizer.convert_ids_to_tokens(ids)
        print(f"    原句: {sentence!r}")
        print(f"    {'token':<14}{'id':>8}")
        for t, i in zip(toks, ids.tolist()):
            print(f"    {t:<14}{i:>8}")
        print(f"\n    特殊标记: [CLS]={tokenizer.cls_token_id}(句子开始)  "
              f"[SEP]={tokenizer.sep_token_id}(句子结束)  [PAD]={tokenizer.pad_token_id}(填充)  "
              f"[MASK]={tokenizer.mask_token_id}  [UNK]={tokenizer.unk_token_id}")
        print(f"    词表大小: {tokenizer.vocab_size}（bert-base-uncased 共 30522 个 token）")
        stats["tokenizer_demo"] = {"sentence": sentence, "tokens": toks,
                                   "ids": ids.tolist(),
                                   "vocab_size": int(tokenizer.vocab_size)}

        section("6) BERT 内部：三层 embedding 相加")
        encoder_txt = TextEncoder(args.text_model, freeze=True, pooling="mean").to(device)
        encoder_txt.eval()
        bert = encoder_txt.bert
        embeddings = bert.embeddings
        print(f"    Token Embedding 矩阵形状: {list(embeddings.word_embeddings.weight.shape)}"
              f"  （词表大小 x 隐藏维度）")
        print(f"    Position Embedding 形状 : {list(embeddings.position_embeddings.weight.shape)}")
        print(f"    Segment Embedding 形状  : {list(embeddings.token_type_embeddings.weight.shape)}")

        ids_d = ids.unsqueeze(0).to(device)
        with torch.no_grad():
            tok_emb = embeddings.word_embeddings(ids_d)
            pos_ids = torch.arange(ids_d.size(1), device=device).unsqueeze(0)
            pos_emb = embeddings.position_embeddings(pos_ids)
            seg_ids = torch.zeros_like(ids_d)
            seg_emb = embeddings.token_type_embeddings(seg_ids)
            total_emb = tok_emb + pos_emb + seg_emb
        print(f"\n    Token Embedding     : {list(tok_emb.shape)}   范数均值={tok_emb.norm(dim=-1).mean():.3f}")
        print(f"    Position Embedding  : {list(pos_emb.shape)}   范数均值={pos_emb.norm(dim=-1).mean():.3f}")
        print(f"    Segment Embedding   : {list(seg_emb.shape)}   范数均值={seg_emb.norm(dim=-1).mean():.3f}")
        print(f"    相加后的输入向量    : {list(total_emb.shape)}   范数均值={total_emb.norm(dim=-1).mean():.3f}")
        print(f"    （这就是公式 最终输入 = Token Embed + Segment Embed + Position Embed）")
        stats["embedding_shapes"] = {
            "token_embedding_matrix": list(embeddings.word_embeddings.weight.shape),
            "position_embedding": list(embeddings.position_embeddings.weight.shape),
            "segment_embedding": list(embeddings.token_type_embeddings.weight.shape),
            "hidden_size": int(bert.config.hidden_size),
        }

        section("7) Transformer 输出与两种池化")
        batch = tokenizer([sentence, "a large brown bird perched on a branch"],
                          padding="max_length", truncation=True,
                          max_length=config.TEXT_MAX_LEN, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = bert(**batch)
        hidden = out.last_hidden_state
        print(f"    BERT 输出 last_hidden_state: {list(hidden.shape)}   (B, L, H)")
        print(f"    L = {hidden.shape[1]}（max_length={config.TEXT_MAX_LEN}），"
              f"H = {hidden.shape[2]}（每个 token 一个 {hidden.shape[2]} 维向量）")

        cls_vec = hidden[:, 0]
        mean_masked = masked_mean(hidden, batch["attention_mask"])
        mean_raw = hidden.mean(dim=1)
        real = int(batch["attention_mask"][0].sum().item())
        print(f"\n    第 1 句真实 token 数 = {real}，其余 {hidden.shape[1] - real} 个是 [PAD]")
        print(f"    [CLS] 向量              : {list(cls_vec.shape)}   范数={cls_vec.norm(dim=-1)[0]:.3f}")
        print(f"    masked mean（正确）     : {list(mean_masked.shape)}   范数={mean_masked.norm(dim=-1)[0]:.3f}")
        print(f"    raw mean（错误示范）    : {list(mean_raw.shape)}   范数={mean_raw.norm(dim=-1)[0]:.3f}")
        dilution = 1 - (mean_raw.norm(dim=-1)[0] / mean_masked.norm(dim=-1)[0]).item()
        cos = torch.nn.functional.cosine_similarity(mean_masked, mean_raw, dim=-1)[0].item()
        print(f"\n    => 不做 mask 会让句向量范数缩水 {dilution:.1%}，"
              f"与正确向量的余弦相似度只有 {cos:.4f}")
        print(f"       （padding 越多、稀释越严重；这就是 attention_mask 必须参与池化的原因）")
        stats["pooling"] = {
            "real_tokens": real, "padded_tokens": int(hidden.shape[1] - real),
            "cls_norm": float(cls_vec.norm(dim=-1)[0]),
            "masked_mean_norm": float(mean_masked.norm(dim=-1)[0]),
            "raw_mean_norm": float(mean_raw.norm(dim=-1)[0]),
            "norm_shrink_ratio": float(dilution),
            "cosine_masked_vs_raw": float(cos),
        }

        section("8) 真实描述的文本特征")
        text_feats, text_labels, texts = [], [], []
        for rec in records:
            enc = tokenizer(rec["captions"][0], padding="max_length", truncation=True,
                            max_length=config.TEXT_MAX_LEN, return_tensors="pt")
            with torch.no_grad():
                v = encoder_txt(enc["input_ids"].to(device),
                                enc["attention_mask"].to(device))
            text_feats.append(v.squeeze(0).cpu().numpy())
            text_labels.append(rec["label"])
            texts.append(rec["captions"][0])
        text_feats = np.stack(text_feats)
        print(f"    文本特征形状: {text_feats.shape}   （每句一个 {text_feats.shape[1]} 维向量）")
        print(f"    图像特征形状: {feats.shape}   （每张图一个 {feats.shape[1]} 维向量）")
        stats["text_feature_dim"] = int(text_feats.shape[1])

        np.savez_compressed(config.FEATURE_ROOT / "text_features.npz",
                            features=text_feats, labels=np.array(text_labels),
                            texts=np.array(texts))

        # ============================================================ 跨模态
        section("9) 跨模态：两个空间根本不可比（这就是为什么需要 CLIP）")
        print(f"    图像向量维度 = {feats.shape[1]}，文本向量维度 = {text_feats.shape[1]}"
              f"  -> 维度都不同，连点积都算不了")
        try:
            _ = feats @ text_feats.T
        except ValueError as exc:
            print(f"    直接算相似度 -> ValueError: {str(exc).splitlines()[0][:70]}...")

        # 为了把"不可比"量化出来，用**随机**投影把两者强行拉到同一维度。
        # 注意：这个投影没有任何训练，所以不可能让两个模态对齐 —— 这正是要展示的。
        rng = np.random.default_rng(config.SEED)
        common = 256
        w_img = rng.normal(0, 1 / np.sqrt(feats.shape[1]), (feats.shape[1], common))
        w_txt = rng.normal(0, 1 / np.sqrt(text_feats.shape[1]), (text_feats.shape[1], common))
        img_p = feats @ w_img
        txt_p = text_feats @ w_txt
        img_p = img_p / np.linalg.norm(img_p, axis=1, keepdims=True)
        txt_p = txt_p / np.linalg.norm(txt_p, axis=1, keepdims=True)

        sim = txt_p @ img_p.T
        diag = np.diag(sim)
        off = sim[~np.eye(len(sim), dtype=bool)]
        print(f"\n    用随机投影强行拉到 {common} 维后再算余弦相似度：")
        print(f"      配对（同一张图的图与文）: 均值={diag.mean():+.4f}  范围=[{diag.min():+.4f}, {diag.max():+.4f}]")
        print(f"      非配对（随机图文）      : 均值={off.mean():+.4f}  范围=[{off.min():+.4f}, {off.max():+.4f}]")
        rank_ok = int((sim.argmax(axis=1) == np.arange(len(sim))).sum())
        print(f"      图文检索命中（对角线最大）: {rank_ok}/{len(sim)}")
        gap = float(diag.mean() - off.mean())
        print(f"\n    => 配对与非配对的相似度差 = {gap:+.4f}（几乎为 0），检索基本等于瞎猜。")
        print(f"       原因：两个编码器各自独立训练，中间没有任何机制要求它们对齐。")
        print(f"       这正是阶段 5 的 CLIP 要做的事 —— 用对比学习把两个模态拉进同一个语义空间。")
        stats["cross_modal"] = {
            "image_dim": int(feats.shape[1]), "text_dim": int(text_feats.shape[1]),
            "projected_dim": common, "projection": "random (untrained)",
            "paired_sim_mean": float(diag.mean()), "unpaired_sim_mean": float(off.mean()),
            "paired_minus_unpaired": gap,
            "retrieval_hits": rank_ok, "num_pairs": int(len(sim)),
        }

        if HAS_MPL:
            section("10) 特征空间可视化")
            # 图像特征自己的分布
            viz.plot_embedding(feats, np.array([r["label"] for r in records]),
                               config.FIG_ROOT / "embedding_image_features.png",
                               class_names=ann.classes,
                               title="图像编码器输出的特征（ResNet18）", method="pca")
            # 两个模态画在同一张图上。
            # 注意：维度不同（512 vs 768），所以先把各自**独立**做 PCA 降到 2 维，
            # 再把两组 2D 点拼在一起 —— 这样才看得见"两个空间是分开的"。
            def _pca2(a):
                a = a - a.mean(axis=0, keepdims=True)
                u, s, _ = np.linalg.svd(a, full_matrices=False)
                return u[:, :2] * s[:2]

            emb_img = _pca2(feats)
            emb_txt = _pca2(text_feats)
            n = len(feats)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7.5, 6))
            ax.scatter(emb_img[:, 0], emb_img[:, 1], s=70, c="#4C72B0", marker="o",
                       label=f"图像特征 [{feats.shape[1]}]")
            ax.scatter(emb_txt[:, 0], emb_txt[:, 1], s=70, c="#C44E52", marker="^",
                       label=f"文本特征 [{text_feats.shape[1]}]")
            for i in range(n):
                ax.plot([emb_img[i, 0], emb_txt[i, 0]], [emb_img[i, 1], emb_txt[i, 1]],
                        c="gray", lw=0.8, alpha=0.7)
            ax.legend(); ax.grid(alpha=0.2)
            ax.set_title("同一张图的图像特征 vs 文本特征（各自 PCA 到 2 维）\n"
                         "灰线 = 同一张图的两个模态，方向随机、长度各异 —— 两个空间没有对齐")
            viz._save(fig, config.FIG_ROOT / "embedding_cross_modal.png")
            viz.plot_similarity_matrix(sim, config.FIG_ROOT / "similarity_before_alignment.png",
                                       row_labels=[f"文本:{t[:16]}" for t in texts],
                                       col_labels=[r["class_name"][:16] for r in records],
                                       title="未对齐时的图文相似度（对角线应最高，实际不是）")
        else:
            print("\n[跳过可视化] 未安装 matplotlib")

        # ============================================================ 落盘
        section("11) 保存分析结果")
        save_json(to_builtin(stats), config.RESULT_ROOT / "03_model_dissection.json")
        print(f"分析结果 -> {config.RESULT_ROOT / '03_model_dissection.json'}")
        print(f"图像特征 -> {config.FEATURE_ROOT / 'image_features.npz'}")
        print(f"文本特征 -> {config.FEATURE_ROOT / 'text_features.npz'}")

        section("小结")
        print("  1. 图像 [3,224,224] --ResNet--> [512,7,7] --pool--> [512] / --flatten--> [49,512]")
        print("  2. 文本 句子 --Tokenizer--> id --BERT--> [L,768] --pool--> [768]")
        print("  3. 两个向量维度不同、空间不同，直接算相似度没有意义")
        print("  4. 所以需要阶段 4 的融合层（把两个模态拼起来做分类）")
        print("     或者阶段 5 的 CLIP 思路（把两个模态对齐到同一个语义空间）")
        print("\n下一步: python 04_multimodal_fusion.py   （多模态融合与消融）")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    sys.exit(main())
