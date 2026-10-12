"""env_check.py —— 环境自检：在跑实验前先确认依赖齐全。

用法
----
    python env_check.py

为什么要这个脚本
----------------
本机有两个 Python：
    D:\\python\\python3.13.15\\python.exe        <-- 没有 torch，跑不了实验
    C:\\Users\\Echo\\.conda\\envs\\dl\\python.exe <-- 装好了 torch/transformers，用这个

如果用了第一个，脚本会报 "No module named 'torch'"。本脚本会把这件事
直白地说清楚，并给出修复命令。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------- 依赖表
REQUIRED = {
    "torch": "深度学习框架",
    "torchvision": "图像模型与预处理",
    "transformers": "BERT / CLIP",
    "PIL": "图像读取（pillow）",
    "numpy": "数值计算",
}
OPTIONAL = {
    "tqdm": "进度条",
    "yaml": "YAML 配置（PyYAML）",
    "matplotlib": "可视化出图",
    "sklearn": "指标计算（scikit-learn）",
}

# 本机已知可用的解释器（按优先级）
KNOWN_INTERPRETERS = [
    r"C:\Users\Echo\.conda\envs\dl\python.exe",
    r"D:\anaconda3\python.exe",
]


def _find(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _version(module: str) -> str:
    try:
        m = __import__(module)
        return getattr(m, "__version__", "?")
    except Exception:
        return "?"


def main() -> int:
    print("=" * 72)
    print("  环境自检")
    print("=" * 72)
    print(f"当前解释器 : {sys.executable}")
    print(f"Python     : {sys.version.split()[0]}")
    print(f"工作目录   : {os.getcwd()}")
    print()

    missing_required, missing_optional = [], []

    print("必需依赖")
    print("-" * 72)
    for mod, desc in REQUIRED.items():
        ok = _find(mod)
        ver = _version(mod) if ok else "-"
        print(f"  [{'OK ' if ok else '缺失'}] {mod:<16}{ver:<16}{desc}")
        if not ok:
            missing_required.append(mod)

    print("\n可选依赖（缺了也能跑，只是少一些图和指标）")
    print("-" * 72)
    for mod, desc in OPTIONAL.items():
        ok = _find(mod)
        ver = _version(mod) if ok else "-"
        print(f"  [{'OK ' if ok else '缺失'}] {mod:<16}{ver:<16}{desc}")
        if not ok:
            missing_optional.append(mod)

    # ---------------------------------------------------------- GPU
    print("\nGPU")
    print("-" * 72)
    if _find("torch"):
        try:
            import torch

            print(f"  CUDA 可用 : {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  设备      : {torch.cuda.get_device_name(0)}")
                print(f"  显存      : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print("  （没有 CUDA，将用 CPU 跑；速度会慢很多，但功能完整）")
        except Exception as exc:
            print(f"  torch 导入失败: {exc}")
    else:
        print("  torch 未安装，无法检查")

    # ---------------------------------------------------------- 数据
    print("\n数据")
    print("-" * 72)
    root = Path(__file__).resolve().parent
    data = root / "data"
    ann = data / "annotations.json"
    images = data / "images"
    print(f"  data/images          : {'存在' if images.is_dir() else '缺失'}"
          f"{f'（{len(list(images.iterdir()))} 个类别）' if images.is_dir() else ''}")
    print(f"  data/annotations.json: {'存在' if ann.exists() else '缺失（先跑 python prepare_data.py）'}")
    if ann.exists():
        try:
            import json

            d = json.loads(ann.read_text(encoding="utf-8"))
            print(f"    类别 {len(d['classes'])} / "
                  f"train {len(d['splits']['train'])} / "
                  f"val {len(d['splits']['val'])} / "
                  f"test {len(d['splits']['test'])}")
        except Exception as exc:
            print(f"    读取失败: {exc}")

    # ---------------------------------------------------------- 模型缓存
    print("\nHuggingFace 模型缓存")
    print("-" * 72)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from birdmm.hub import describe_cache, is_cached

        print("  " + describe_cache().replace("\n", "\n  "))
        for name in ("bert-base-uncased", "openai/clip-vit-base-patch32"):
            mark = "已缓存" if is_cached(name) else "未缓存（首次运行需联网下载）"
            print(f"  [{mark}] {name}")
        print("\n  提示：已缓存的模型即使断网也能用（代码里做了离线回退）")
    except Exception as exc:
        print(f"  检查失败: {exc}")

    # ---------------------------------------------------------- 结论
    print("\n" + "=" * 72)
    if not missing_required:
        print("  结论：环境就绪，可以直接开始实验")
        print("=" * 72)
        print("\n下一步：")
        print("  python run_all.py --dry-run     # 看计划")
        print("  python run_all.py               # 全部跑完")
        return 0

    print("  结论：环境不完整，缺少必需依赖")
    print("=" * 72)
    print(f"\n缺少：{', '.join(missing_required)}")

    # 帮忙找找本机有没有别的能用的解释器
    print("\n本机可用的其他解释器：")
    usable = []
    for p in KNOWN_INTERPRETERS:
        if Path(p).exists() and Path(p).resolve() != Path(sys.executable).resolve():
            usable.append(p)
            print(f"  {p}")
    if not usable:
        print("  （没找到已知的可用解释器）")

    print("\n" + "-" * 72)
    print("修复办法（二选一）")
    print("-" * 72)
    if usable:
        print("【推荐】换用已装好依赖的解释器 —— 本机已经有一个：")
        print()
        print(f'  & "{usable[0]}" prepare_data.py')
        print()
        print("  在 VS Code 里永久切换：")
        print("    Ctrl+Shift+P  ->  输入 'Python: Select Interpreter'  ->  选")
        print(f"    {usable[0]}")
        print()
        print("  也可以直接看本工程 .vscode/settings.json（已经预设好这个路径）")
        print()
        print("【或者】给当前解释器装依赖：")
    else:
        print("给当前解释器装依赖：")
    print()
    print(f'  & "{sys.executable}" -m pip install -r requirements.txt '
          f'-i https://pypi.tuna.tsinghua.edu.cn/simple')
    print("  # torch 请按 https://pytorch.org 的命令装与你 CUDA 匹配的版本")
    return 1


if __name__ == "__main__":
    sys.exit(main())
