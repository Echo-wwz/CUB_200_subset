"""utils.py —— 通用工具：随机种子、日志、计时、JSON 落盘、可选依赖。"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------ 随机种子
def set_seed(seed: int = 42) -> None:
    """固定所有随机源，保证多次运行结果一致。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cudnn 的 benchmark 会让同一份代码每次跑出不同结果，实验对比时关掉
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ------------------------------------------------------------------ 日志
class Tee:
    """把 print 同时写到终端和文件。

    写文件时显式用 UTF-8，这样重定向到管道 / 文件都不会因为 GBK 代码页崩掉。
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data: str) -> None:
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        try:
            sys.stdout = self.stdout
        finally:
            self.file.close()

    def isatty(self) -> bool:            # tqdm 等库会问这个
        return getattr(self.stdout, "isatty", lambda: False)()

    def fileno(self):
        return self.stdout.fileno()

    @property
    def encoding(self):
        return getattr(self.stdout, "encoding", "utf-8")


def log_to_file(path: Path) -> Tee:
    """开始把 stdout 复制到日志文件；返回的 Tee 需要调用 .close() 还原。"""
    tee = Tee(Path(path))
    sys.stdout = tee
    return tee


def banner(title: str, width: int = 78) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def section(title: str, width: int = 78) -> None:
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


# ------------------------------------------------------------------ 计时
class Timer:
    """with Timer('训练') as t: ...  ->  t.elapsed"""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0
        if self.name:
            print(f"[耗时] {self.name}: {format_time(self.elapsed)}")


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


# ------------------------------------------------------------------ JSON
def save_json(obj, path: Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_builtin(obj):
    """把 numpy / torch 标量递归转成原生 Python 类型，方便 json.dump。"""
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if hasattr(obj, "tolist"):          # numpy / torch tensor
        return obj.tolist()
    if hasattr(obj, "item"):            # numpy / torch 标量
        return obj.item()
    return obj


# ------------------------------------------------------------------ 可选依赖
def has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def try_import(name: str):
    """可选依赖：装了就用，没装返回 None（这些库不装也能跑）。"""
    try:
        return __import__(name)
    except Exception:
        return None


HAS_SKLEARN = has_module("sklearn")
HAS_MPL = has_module("matplotlib")
HAS_TQDM = has_module("tqdm")


def progress(iterable, **kwargs):
    """有 tqdm 就用 tqdm，没有就原样返回。"""
    if HAS_TQDM:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    return iterable


# ------------------------------------------------------------------ 参数量
def count_parameters(model) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def human_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
