"""text_encoder.py —— 文本编码器：把一句话变成一个向量。

设计要点
--------
1. 分词：BERT tokenizer 把句子切成 WordPiece token，加上 [CLS]/[SEP]，
   每个 token 映射为一个 id；id 再查 embedding 表得到初始向量。
2. 池化：BERT 输出每个 token 的上下文向量 [B, L, 768]，
   我们需要"整句向量"，两种常用做法：
   - cls  ：直接取 [CLS] 位置的向量；
   - mean ：对所有真实 token（按 attention_mask 过滤 padding）取平均 —— 更稳。
3. freeze=True 时只训练融合层：BERT 完全不动，速度和显存都友好。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel, AutoTokenizer

from .hub import load_pretrained


def load_tokenizer(model_name: str = "bert-base-uncased"):
    """加载 tokenizer。国内网络不通时可先设置镜像：
    PowerShell:  $env:HF_ENDPOINT="https://hf-mirror.com"
    CMD:         set HF_ENDPOINT=https://hf-mirror.com

    联网失败时会自动退回本地缓存（见 birdmm/hub.py），
    所以只要下载过一次，之后断网也能跑。
    """
    return load_pretrained(AutoTokenizer.from_pretrained, model_name)


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """只对真实 token 求平均（把 padding 位置屏蔽掉）。

    hidden: [B, L, H]；attention_mask: [B, L]（1=真实 token，0=padding）

    为什么必须过滤 [PAD]：一句话只有 5 个真实单词却补了 20 个 [PAD] 时，
    直接对 25 个向量求平均会把真正有意义的向量严重稀释（平均值被拉向零）。
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)      # [B, L, 1]
    summed = (hidden * mask).sum(dim=1)                       # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-6)                  # [B, 1]
    return summed / counts


class TextEncoder(nn.Module):
    """BERT 文本编码器。

    Args:
        model_name: HuggingFace 模型名，如 bert-base-uncased / distilbert-base-uncased
        freeze:     是否冻结 BERT 主干
        out_dim:    输出维度。None 表示保持 BERT 原生维度（bert-base = 768）
        pooling:    "cls" 或 "mean"
        dropout:    投影层 dropout
    """

    def __init__(self, model_name: str = "bert-base-uncased", pretrained: bool = True,
                 freeze: bool = False, out_dim: int | None = None,
                 pooling: str = "mean", dropout: float = 0.0):
        super().__init__()
        if pretrained:
            # 带离线回退：网络抖动时用本地缓存，不会让整个实验崩掉
            self.bert = load_pretrained(AutoModel.from_pretrained, model_name)
        else:
            config = load_pretrained(AutoConfig.from_pretrained, model_name)
            self.bert = AutoModel.from_config(config)
        self.hidden_size = self.bert.config.hidden_size   # bert-base -> 768
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling 只能是 'cls' 或 'mean'，收到：{pooling}")
        self.pooling = pooling
        self.out_dim = self.hidden_size if out_dim is None else out_dim
        if out_dim is None:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.hidden_size, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        self.freeze = freeze
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "TextEncoder":
        super().train(mode)
        if self.freeze:
            self.bert.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                return_tokens: bool = False):
        """
        Args:
            return_tokens: False -> [B, out_dim]（整句向量）
                           True  -> (整句向量, token 序列 [B, L, out_dim])
                                    token 序列给 Cross-Attention 融合当 Key/Value
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                     # [B, L, H]
        if self.pooling == "cls":
            pooled = hidden[:, 0]                          # [CLS] 位置
        else:
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                pooled = masked_mean(hidden, attention_mask)
        sentence = self.proj(pooled)                       # [B, out_dim]
        if return_tokens:
            return sentence, self.proj(hidden)             # [B, L, out_dim]
        return sentence

    def extra_repr(self) -> str:
        return (f"hidden_size={self.hidden_size}, out_dim={self.out_dim}, "
                f"pooling={self.pooling}, frozen={'(bert)' if self.freeze else 'no'}")


class TextClassifier(nn.Module):
    """文本单模态分类器 = BERT 编码器 + 线性分类头。"""

    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 100,
                 pretrained: bool = True, freeze: bool = False, pooling: str = "mean",
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = TextEncoder(model_name, pretrained=pretrained, freeze=freeze,
                                   pooling=pooling)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        feat = self.encoder(input_ids, attention_mask)
        return self.classifier(self.dropout(feat))

    def features(self, input_ids, attention_mask=None):
        return self.encoder(input_ids, attention_mask)
