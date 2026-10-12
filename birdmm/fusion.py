"""fusion.py —— 多模态融合模块（本实验的核心）。

1. Early Fusion (ConcatFusion)
      [图像向量, 文本向量] -> 拼接 -> MLP -> 分类
   优点：简单、参数少；缺点：两个模态从头到尾被当成一串数字，没有交互。

2. Late Fusion (LateFusion)
      图像 -> 图像分类头 -> 概率
      文本 -> 文本分类头 -> 概率
      两者加权平均 -> 最终概率
   优点：任一模态缺失都能工作、可解释；缺点：模态之间从不交流。

3. Cross-Attention Fusion (CrossAttentionFusion)  ← 推荐
      图像 token 作为 Query，文本 token 作为 Key/Value，
      让"图像的每个空间位置"主动去文本里找相关描述（"这里是不是 blue？"）。
   优点：显式建模跨模态对齐；可导出注意力权重，用于可视化"模型在看哪个词"。

4. Gated Fusion (GatedFusion)
      用一个门控网络逐个样本决定"这次更信图像还是更信文本"。
   优点：对"描述没提颜色"这类不可靠文本有天然的抑制能力。

所有融合模块统一返回 dict：
    {"logits": [B, C],
     "image_logits": [...]?,   # 仅 Late Fusion 有
     "text_logits": [...]?,    # 仅 Late Fusion 有
     "attn_weights": [...]?,   # 仅 Attention Fusion 有，用于可视化
     "gate": [...]?}           # 仅 Gated Fusion 有

因为接口完全一致，换融合方式只需要改一个字符串 —— 这是消融实验能做得干净的前提。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

FUSION_TYPES = ("early", "late", "attention", "gated")
FUSION_ALIASES = {
    "early": "early", "concat": "early",
    "late": "late",
    "attention": "attention", "cross": "attention", "cross_attention": "attention", "attn": "attention",
    "gated": "gated", "gate": "gated",
}


def normalize_fusion_type(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    if key not in FUSION_ALIASES:
        raise ValueError(f"未知融合方式 '{name}'，可选：{FUSION_TYPES}")
    return FUSION_ALIASES[key]


class _Projection(nn.Module):
    """可选的公共维度投影：把两个模态压到同一个 d，便于控制参数量。"""

    def __init__(self, image_dim: int, text_dim: int, common_dim: Optional[int]):
        super().__init__()
        if common_dim is None:
            self.image_proj = nn.Identity()
            self.text_proj = nn.Identity()
            self.image_dim, self.text_dim = image_dim, text_dim
        else:
            self.image_proj = nn.Linear(image_dim, common_dim)
            self.text_proj = nn.Linear(text_dim, common_dim)
            self.image_dim = self.text_dim = common_dim

    def forward(self, image_feat, text_feat):
        return self.image_proj(image_feat), self.text_proj(text_feat)


# ==================================================================== ① Early
class ConcatFusion(nn.Module):
    """Early Fusion：拼接 + MLP。

    图像[512] ┐
              ├─ concat ─► [1280] ─► MLP ─► 类别
    文本[768] ┘
    """

    def __init__(self, image_dim: int = 512, text_dim: int = 768, num_classes: int = 100,
                 hidden_dim: int = 512, dropout: float = 0.3, common_dim: Optional[int] = None):
        super().__init__()
        self.fusion_type = "early"
        self.proj = _Projection(image_dim, text_dim, common_dim)
        in_dim = self.proj.image_dim + self.proj.text_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.in_dim = in_dim

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                text_tokens: torch.Tensor | None = None,
                image_tokens: torch.Tensor | None = None, **kwargs) -> Dict[str, torch.Tensor]:
        image_feat, text_feat = self.proj(image_feat, text_feat)
        fused = torch.cat([image_feat, text_feat], dim=-1)          # [B, 1280]
        return {"logits": self.mlp(fused), "fused": fused}


# ==================================================================== ② Late
class LateFusion(nn.Module):
    """Late Fusion：两个模态各自出概率，再加权平均。

        image ─► 图像分类头 ─► p_img  ┐
                                      ├─ w·p_text + (1-w)·p_img ─► 类别
        text  ─► 文本分类头 ─► p_text ┘

    w 可学习（sigmoid 参数化，初值 0.5）；也可用 learnable_weight=False 固定。
    返回 image_logits / text_logits，所以"只给图像"或"只给文本"时照样能预测。
    """

    def __init__(self, image_dim: int = 512, text_dim: int = 768, num_classes: int = 100,
                 hidden_dim: int = 512, dropout: float = 0.3, common_dim: Optional[int] = None,
                 learnable_weight: bool = True, init_weight: float = 0.5):
        super().__init__()
        self.fusion_type = "late"
        self.proj = _Projection(image_dim, text_dim, common_dim)

        self.image_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.proj.image_dim, num_classes))
        self.text_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.proj.text_dim, num_classes))

        self.learnable_weight = learnable_weight
        if learnable_weight:
            # logit(w)：sigmoid 之后落在 (0,1)，初值 w=0.5 -> logit=0
            init_w = min(max(init_weight, 1e-4), 1 - 1e-4)
            self.w_logit = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(
                init_w / (1 - init_w))))))
        else:
            self.register_buffer("w_fixed", torch.tensor(float(init_weight)))

    @property
    def weight(self) -> torch.Tensor:
        """文本模态的权重 w（1-w 给图像）。"""
        if self.learnable_weight:
            return torch.sigmoid(self.w_logit)
        return self.get_buffer("w_fixed")       # register_buffer 注册的 Tensor

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                text_tokens: torch.Tensor | None = None,
                image_tokens: torch.Tensor | None = None,
                image_only: bool = False, text_only: bool = False,
                **kwargs) -> Dict[str, torch.Tensor]:
        image_feat, text_feat = self.proj(image_feat, text_feat)
        image_logits = self.image_head(image_feat)
        text_logits = self.text_head(text_feat)

        if image_only:
            return {"logits": image_logits, "image_logits": image_logits,
                    "text_logits": text_logits, "weight": self.weight.detach()}
        if text_only:
            return {"logits": text_logits, "image_logits": image_logits,
                    "text_logits": text_logits, "weight": self.weight.detach()}

        w = self.weight
        logits = w * text_logits + (1 - w) * image_logits
        return {"logits": logits, "image_logits": image_logits,
                "text_logits": text_logits, "weight": w}


# ============================================================ ③ Cross-Attention
class CrossAttentionFusion(nn.Module):
    """Cross-Attention Fusion：图像 token 做 Query，文本 token 做 Key/Value。

        image tokens [49, 512] ── Query ┐
                                        ├─ Multi-Head Attention ─► [49, d] ─► 池化 ─► 分类
        text  tokens [L, 768]  ── Key/Value ┘

    让"图像的每个位置"去文本里找相关描述（"我这里是 blue 吗？"）。
    注意力权重 [B, heads, 49, L] 会被返回，可做可视化看"模型在看哪个词"。
    """

    def __init__(self, image_dim: int = 512, text_dim: int = 768, num_classes: int = 100,
                 hidden_dim: int = 512, dropout: float = 0.3, common_dim: Optional[int] = None,
                 num_heads: int = 4, return_attn: bool = True):
        super().__init__()
        self.fusion_type = "attention"
        self.proj = _Projection(image_dim, text_dim, common_dim)
        self.d_model = self.proj.image_dim            # Query 与输出都在这个维度上
        assert self.d_model % num_heads == 0, \
            f"d_model({self.d_model}) 必须能被 num_heads({num_heads}) 整除"

        # 文本侧也要投到同一维度才能算注意力（kdim/vdim 允许与 embed_dim 不同）
        self.text_key_proj = (nn.Identity() if self.proj.text_dim == self.d_model
                              else nn.Linear(self.proj.text_dim, self.d_model))
        self.text_value_proj = (nn.Identity() if self.proj.text_dim == self.d_model
                                else nn.Linear(self.proj.text_dim, self.d_model))

        self.attn = nn.MultiheadAttention(self.d_model, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, self.d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model), nn.Linear(self.d_model, num_classes))
        self.return_attn = return_attn

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                text_tokens: torch.Tensor | None = None,
                image_tokens: torch.Tensor | None = None,
                return_attn: bool | None = None, **kwargs) -> Dict[str, torch.Tensor]:
        if image_tokens is None or text_tokens is None:
            raise ValueError("CrossAttentionFusion 需要 image_tokens 与 text_tokens，"
                             "请用 encoder(..., return_tokens=True) 取 token 序列")

        q = self.proj.image_proj(image_tokens)                    # [B, 49, d]
        t = self.proj.text_proj(text_tokens)                      # [B, L, d]
        k = self.text_key_proj(t)                                 # [B, L, d]
        v = self.text_value_proj(t)                               # [B, L, d]

        # nn.MultiheadAttention 的 key_padding_mask：True 表示"忽略这个位置"
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)              # [B, L]

        want_attn = self.return_attn if return_attn is None else return_attn
        attn_out, attn_w = self.attn(
            q, k, v,
            key_padding_mask=key_padding_mask,
            need_weights=want_attn,
            average_attn_weights=False,                           # 保留每个 head，便于可视化
        )
        x = self.norm1(q + self.dropout(attn_out))                # 残差 + LayerNorm
        x = self.norm2(x + self.dropout(self.ffn(x)))
        pooled = x.mean(dim=1)                                    # 49 个 token 平均池化
        out = {"logits": self.classifier(pooled), "fused": pooled}
        if want_attn and attn_w is not None:
            out["attn_weights"] = attn_w                          # [B, heads, 49, L]
        return out


# ==================================================================== ④ Gated
class GatedFusion(nn.Module):
    """Gated Fusion：逐样本决定"这次更信图像还是更信文本"。

        a = σ(W[img; txt])      fusion = a·img + (1-a)·txt

    对"描述没提颜色"这类不可靠文本有天然的抑制能力。
    gate 会被返回，可以统计它在不同类别/不同描述质量下的分布。
    """

    def __init__(self, image_dim: int = 512, text_dim: int = 768, num_classes: int = 100,
                 hidden_dim: int = 512, dropout: float = 0.3, common_dim: Optional[int] = None,
                 gate_hidden: int = 256):
        super().__init__()
        self.fusion_type = "gated"
        self.proj = _Projection(image_dim, text_dim, common_dim)
        d_img, d_txt = self.proj.image_dim, self.proj.text_dim

        self.gate_net = nn.Sequential(
            nn.Linear(d_img + d_txt, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, d_img),
            nn.Sigmoid(),                                          # a ∈ (0,1)，逐维门控
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_img),
            nn.Dropout(dropout),
            nn.Linear(d_img, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                text_tokens: torch.Tensor | None = None,
                image_tokens: torch.Tensor | None = None,
                text_only: bool = False, image_only: bool = False,
                **kwargs) -> Dict[str, torch.Tensor]:
        image_feat, text_feat = self.proj(image_feat, text_feat)

        if text_only:
            # 模态缺失时退化为"只用文本"：把 gate 固定成 0
            gate = torch.zeros_like(image_feat)
        elif image_only:
            gate = torch.ones_like(image_feat)
        else:
            gate = self.gate_net(torch.cat([image_feat, text_feat], dim=-1))

        fused = gate * image_feat + (1 - gate) * text_feat
        return {"logits": self.classifier(fused), "gate": gate, "fused": fused}


# ==================================================================== 工厂
def build_fusion(name: str, image_dim: int = 512, text_dim: int = 768,
                 num_classes: int = 100, hidden_dim: int = 512, dropout: float = 0.3,
                 common_dim: Optional[int] = None, num_heads: int = 4,
                 **kwargs) -> nn.Module:
    """按名字造融合模块 —— 消融实验只改这一个字符串。"""
    kind = normalize_fusion_type(name)
    common = dict(image_dim=image_dim, text_dim=text_dim, num_classes=num_classes,
                  hidden_dim=hidden_dim, dropout=dropout, common_dim=common_dim)
    if kind == "early":
        return ConcatFusion(**common, **kwargs)
    if kind == "late":
        return LateFusion(**common, **kwargs)
    if kind == "attention":
        return CrossAttentionFusion(**common, num_heads=num_heads, **kwargs)
    if kind == "gated":
        return GatedFusion(**common, **kwargs)
    raise ValueError(kind)


FUSION_DESCRIPTIONS = {
    "early": "Early Fusion（拼接+MLP）：最简单，但两模态没有显式交互",
    "late": "Late Fusion（双头加权）：可解释、抗模态缺失，但模态之间从不交流",
    "attention": "Cross-Attention Fusion（图像 Q × 文本 KV）：显式跨模态对齐，可导出注意力",
    "gated": "Gated Fusion（逐样本门控）：自动决定更信图像还是更信文本",
}
