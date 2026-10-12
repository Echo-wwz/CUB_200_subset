"""birdmm —— 鸟类多模态分类实验的共享代码包。

模块划分
--------
utils.py          通用工具：随机种子、日志、计时、JSON 落盘、可选依赖
metrics.py        分类指标：Accuracy / Precision / Recall / F1 / Top-5 / 混淆矩阵
data.py           数据集：ImageDataset / TextDataset / MultimodalDataset
image_encoder.py  图像编码器（ResNet / MobileNetV2）
text_encoder.py   文本编码器（BERT）
fusion.py         四种多模态融合模块（Early / Late / Cross-Attention / Gated）
viz.py            可视化：训练曲线、混淆矩阵、特征图、注意力热力图
"""
from __future__ import annotations

import os as _os
import sys as _sys


def _bootstrap_console() -> None:
    """Windows 控制台兼容处理。

    * 终端里直接运行时，Python 走 WriteConsoleW，中文原生显示，无需处理；
    * 但把输出重定向到文件（python x.py > log.txt）或管道时，Python 会按系统
      代码页（简体中文 Windows 是 GBK）编码，遇到 "→" "★" 这类字符会直接
      UnicodeEncodeError 崩溃。这里统一改成 UTF-8 并容错。
    """
    for stream in (_sys.stdout, _sys.stderr):
        try:
            if stream is None or stream.isatty():
                continue
            # reconfigure 只在 TextIOWrapper 上有；用 getattr 保证类型安全
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _quiet_libraries() -> None:
    """关掉第三方库的加载进度条与冗余报告，让实验输出干净可读。"""
    _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    _os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    _os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity_error()
    except Exception:
        pass


class MissingDependency(RuntimeError):
    """缺少必需依赖时抛出，附带可直接照做的修复命令。"""


# 本机已知装好依赖的解释器（按优先级）
_KNOWN_INTERPRETERS = [
    r"C:\Users\Echo\.conda\envs\dl\python.exe",
    r"D:\anaconda3\python.exe",
]


def require(*modules: str) -> None:
    """检查必需依赖，缺失时给出**可照做**的提示，而不是一句 ModuleNotFoundError。

    用法（在脚本开头、import torch 之前调用）：
        from birdmm import require
        require("torch", "torchvision", "transformers")
    """
    import importlib.util
    from pathlib import Path

    missing = []
    for mod in modules:
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError):
            missing.append(mod)
    if not missing:
        return

    lines = [
        "",
        "=" * 74,
        "  缺少必需的依赖，无法运行",
        "=" * 74,
        f"  当前解释器 : {_sys.executable}",
        f"  缺少的包   : {', '.join(missing)}",
        "",
        "  修复办法（二选一）",
        "-" * 74,
    ]

    usable = [p for p in _KNOWN_INTERPRETERS
              if Path(p).exists() and str(Path(p).resolve()) != str(Path(_sys.executable).resolve())]
    if usable:
        lines += [
            "  【推荐】换用本机已装好依赖的解释器：",
            "",
            f'      & "{usable[0]}" ' + Path(_sys.argv[0]).name,
            "",
            "    VS Code 里永久切换：",
            "      Ctrl+Shift+P -> Python: Select Interpreter -> 选",
            f"      {usable[0]}",
            "",
            "  【或者】给当前解释器装依赖：",
            "",
        ]
    lines += [
        f'      & "{_sys.executable}" -m pip install -r requirements.txt '
        f'-i https://pypi.tuna.tsinghua.edu.cn/simple',
        "      # torch 请按 https://pytorch.org 的命令装与你 CUDA 匹配的版本",
        "",
        "  也可以先跑环境自检：python env_check.py",
        "=" * 74,
        "",
    ]
    raise MissingDependency("\n".join(lines))


_bootstrap_console()
_quiet_libraries()


def __getattr__(name: str):
    """惰性导入子模块：`from birdmm import viz` 仍然可用，
    但不会在 import birdmm 时就把 torch / matplotlib 全部拉起来
    （依赖检查 require() 需要在没有 torch 时也能先跑）。"""
    import importlib

    _submodules = {
        "utils", "metrics", "data", "image_encoder", "text_encoder",
        "fusion", "engine", "hub", "viz",
    }
    if name in _submodules:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module          # 缓存，后续访问不再走 __getattr__
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "utils",
    "metrics",
    "data",
    "image_encoder",
    "text_encoder",
    "fusion",
    "engine",
    "hub",
    "viz",
    "require",
    "MissingDependency",
]
