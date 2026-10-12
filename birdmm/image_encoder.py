"""image_encoder.py —— 图像编码器：把一张图变成一个向量（或一组 token）。

设计要点
--------
1. ResNet 的最后两层 avgpool + fc 被去掉后，输出的是 [B, C, h, w] 特征图
   （224 输入 -> [B, 512, 7, 7]）。它同时支持两种用法：
      * 池化成一个全局向量 [B, 512]        -> Early / Late 融合用；
      * 展平成 49 个 token [B, 49, 512]    -> Attention 融合用
        （把图像的每个空间位置当成一个"视觉词"，才能和文本 token 做注意力交互）
2. freeze=True 时主干不训练：只训练融合层 + 分类头，
   CPU 上也能在几分钟内跑完一轮，适合快速迭代。
3. 展平的两种等价写法（结果完全一样，任选其一）：
      fmap.flatten(2).transpose(1, 2)          # [B,C,h,w] -> [B,C,hw] -> [B,hw,C]
      fmap.reshape(B, C, -1).permute(0, 2, 1)  # 同上
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# backbone 名称 -> (构造函数名, ImageNet 预训练权重枚举名)
_ZOO = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet34": ("resnet34", "ResNet34_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights"),
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
}


def _build_backbone(backbone: str, pretrained: bool):
    """兼容新旧 torchvision 的预训练参数写法。"""
    if backbone not in _ZOO:
        raise ValueError(f"目前支持 {list(_ZOO)}，收到：{backbone}")
    ctor_name, weights_name = _ZOO[backbone]
    ctor = getattr(models, ctor_name)
    try:  # torchvision >= 0.13
        weights_cls = getattr(models, weights_name, None)
        weights = None
        if pretrained and weights_cls is not None:
            weights = weights_cls.IMAGENET1K_V1
        return ctor(weights=weights)
    except TypeError:  # 老版本 torchvision
        return ctor(pretrained=pretrained)


class ImageEncoder(nn.Module):
    """ResNet / MobileNetV2 图像编码器。

    Args:
        backbone:   resnet18 / resnet34 / resnet50 / mobilenet_v2
        pretrained: 是否加载 ImageNet 预训练权重（强烈建议 True）
        freeze:     是否冻结主干（只训练后面的投影层与融合层）
        out_dim:    输出维度。None 表示保持主干原生维度（resnet18 -> 512）
        dropout:    投影层 dropout
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, freeze: bool = False,
                 out_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        net = _build_backbone(backbone, pretrained)
        self.backbone = backbone
        self.feature_dim = self._feature_dim(net, backbone)

        # 去掉最后的池化层与全连接层，只保留到最后一层卷积 -> [B, C, h, w]
        if backbone.startswith("resnet"):
            self.body = nn.Sequential(*list(net.children())[:-2])
        else:  # mobilenet_v2 / mobilenet_v3：features 之后就是池化层
            self.body = net.features

        self.out_dim = self.feature_dim if out_dim is None else out_dim
        self.pool = nn.AdaptiveAvgPool2d(1)

        if out_dim is None:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.feature_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.freeze = freeze
        if freeze:
            for p in self.body.parameters():
                p.requires_grad = False

    @staticmethod
    def _feature_dim(net, backbone: str) -> int:
        if backbone.startswith("resnet"):
            return net.fc.in_features                       # resnet18 -> 512, resnet50 -> 2048
        last_conv = net.features[-1]
        # mobilenet_v2 最后是 ConvBNActivation(0=Conv2d)
        conv = last_conv if isinstance(last_conv, nn.Conv2d) else last_conv[0]
        return conv.out_channels                            # mobilenet_v2 -> 1280

    def train(self, mode: bool = True) -> "ImageEncoder":
        """冻结时把主干固定在 eval 模式：BatchNorm 的 running stats 不再更新，
        否则即使 requires_grad=False，BN 仍会在训练中被改写，导致"冻结"名不副实。"""
        super().train(mode)
        if self.freeze:
            self.body.eval()
        return self

    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]，已经过 ImageNet 标准化
            return_tokens: True -> [B, h*w, out_dim]（空间 token）
                           False -> [B, out_dim]（全局特征）
        """
        fmap = self.body(x)                                  # [B, C, h, w]
        if return_tokens:
            tokens = fmap.flatten(2).transpose(1, 2)         # [B, h*w, C]
            return self.proj(tokens)                         # Linear 作用在最后一维
        pooled = self.pool(fmap).flatten(1)                  # [B, C]
        return self.proj(pooled)

    def forward_maps(self, x: torch.Tensor):
        """返回 (全局向量, 空间特征图)，供模型拆解脚本可视化用。"""
        fmap = self.body(x)                                  # [B, C, h, w]
        pooled = self.pool(fmap).flatten(1)
        return self.proj(pooled), fmap

    def extra_repr(self) -> str:
        return (f"backbone={self.backbone}, feature_dim={self.feature_dim}, "
                f"out_dim={self.out_dim}, frozen={self.freeze}")


class ImageClassifier(nn.Module):
    """图像单模态分类器 = 编码器 + 线性分类头。

    CNN 分类器的常规做法：对 512 维特征接一个全连接层（FC），得到类别概率。
    """

    def __init__(self, backbone: str = "resnet18", num_classes: int = 100,
                 pretrained: bool = True, freeze: bool = True, dropout: float = 0.2):
        super().__init__()
        self.encoder = ImageEncoder(backbone, pretrained=pretrained, freeze=freeze)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)              # [B, 512]
        return self.classifier(self.dropout(feat))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """只取编码后的特征向量（模型拆解用）。"""
        return self.encoder(x)
