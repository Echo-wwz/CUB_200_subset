"""config.py —— 全实验统一配置。

所有脚本都从这里读路径与超参数，改一处即可全局生效。
路径全部基于本文件所在目录解析，因此 **在哪个目录下运行都不影响**。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "data"           # 数据根目录
IMAGE_ROOT = DATA_ROOT / "images"           # data/images/<类别>/*.jpg
TEXT_ROOT = DATA_ROOT / "text"              # data/text/<类别>/*.txt（与 jpg 同名）
ANNOTATIONS = DATA_ROOT / "annotations.json"  # 标注缓存（6:2:2 划分被冻结在这里）

OUTPUT_ROOT = PROJECT_ROOT / "outputs"      # 所有结果落盘处
CKPT_ROOT = OUTPUT_ROOT / "checkpoints"
LOG_ROOT = OUTPUT_ROOT / "logs"
FIG_ROOT = OUTPUT_ROOT / "figures"
RESULT_ROOT = OUTPUT_ROOT / "results"
FEATURE_ROOT = OUTPUT_ROOT / "features"     # 提取出的特征向量（.npz）

for _d in (OUTPUT_ROOT, CKPT_ROOT, LOG_ROOT, FIG_ROOT, RESULT_ROOT, FEATURE_ROOT):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 数据
SEED = 42                                   # 固定随机种子，保证划分/训练可复现
SPLIT_RATIO = (0.6, 0.2, 0.2)               # 训练:验证:测试 = 6:2:2
NUM_CLASSES = 100
NUM_CAPTIONS = 10                           # 每张图对应 10 条描述

# ---------------------------------------------------------------- 图像
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------- 文本
BERT_NAME = "bert-base-uncased"
TEXT_MAX_LEN = 48                           # CUB 描述都很短（10~20 词），48 足够
CLIP_NAME = "openai/clip-vit-base-patch32"

# ---------------------------------------------------------------- 训练超参
IMG_BACKBONE = "resnet18"                   # resnet18 / resnet34 / resnet50 / mobilenet_v2
IMG_BATCH_SIZE = 32
IMG_EPOCHS = 8
IMG_LR = 1e-3                               # 冻结主干时用 1e-3
IMG_LR_FINETUNE = 1e-4                      # 解冻主干时用 1e-4
IMG_FREEZE = True                           # True = 只训练分类头（快，CPU 也能跑）

TXT_BATCH_SIZE = 32
TXT_EPOCHS = 8
TXT_LR = 2e-5
TXT_FREEZE = False                          # BERT 微调（GPU 上很快）
TXT_POOLING = "mean"                        # cls / mean

FUSION_BATCH_SIZE = 32
FUSION_EPOCHS = 8
FUSION_LR = 1e-3
FUSION_FREEZE = True                        # 冻结两个主干，只训练融合层 —— 消融才干净
FUSION_DIM = 256                            # 融合空间的公共维度 d

# ---------------------------------------------------------------- 运行环境
NUM_WORKERS = 0                             # Windows 下 DataLoader 多进程容易出问题，默认 0
DEVICE = None                               # 由 get_device() 决定，见下


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def get_device(prefer: str | None = None):
    """返回 torch.device。prefer 可为 'cuda' / 'cpu' / None（自动）。"""
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hf_endpoint() -> str | None:
    """HuggingFace 镜像（国内网络不通时用）。设置环境变量 HF_ENDPOINT 即可生效。"""
    return os.environ.get("HF_ENDPOINT")


def describe() -> str:
    lines = [
        f"项目根目录 : {PROJECT_ROOT}",
        f"数据根目录 : {DATA_ROOT}",
        f"输出目录   : {OUTPUT_ROOT}",
        f"图像主干   : {IMG_BACKBONE}  (freeze={IMG_FREEZE})",
        f"文本主干   : {BERT_NAME}  (pooling={TXT_POOLING}, freeze={TXT_FREEZE})",
        f"CLIP       : {CLIP_NAME}",
    ]
    try:
        import torch

        lines.append(f"torch      : {torch.__version__}  cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"GPU        : {torch.cuda.get_device_name(0)}")
    except Exception:
        # 这里只报告事实，并明确说清楚"哪些脚本还能跑、哪些不能"，
        # 免得用户看到一行"未安装"就以为整个工程坏了。
        import sys

        lines.append(f"torch      : 未安装（当前解释器 {sys.executable}）")
        lines.append("             本脚本（数据准备）不需要 torch，可以继续；")
        lines.append("             但 01~05 的训练/评估脚本需要，请换用装好依赖的解释器：")
        lines.append(r"             C:\Users\Echo\.conda\envs\dl\python.exe")
        lines.append("             详见：python env_check.py")
    return "\n".join(lines)
