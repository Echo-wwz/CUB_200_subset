"""hub.py —— 稳健的 HuggingFace 模型加载。

问题
----
`from_pretrained("bert-base-uncased")` 默认会联网向 HuggingFace 查询最新版本。
一旦网络不通（国内很常见，或者只是临时抖动），它会抛
    httpx.ConnectTimeout: [WinError 10060] ...
**即使本地缓存里模型是完整的**，脚本也会直接崩掉。

解决办法
--------
先按正常方式联网加载；失败时自动退回 `local_files_only=True` 用本地缓存。
这样"第一次要联网下载、之后永远离线可用"，网络抖动不再影响实验。

用法
----
    from birdmm.hub import load_pretrained
    tok = load_pretrained(AutoTokenizer.from_pretrained, "bert-base-uncased")
    model = load_pretrained(AutoModel.from_pretrained, "bert-base-uncased")
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

# 缩短联网等待时间：默认值在断网时要等好几分钟才报错
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

_warned: set[str] = set()


def cache_dir() -> Path:
    """HuggingFace 缓存目录。"""
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_cached(model_name: str) -> bool:
    """粗略判断某个模型是否已在本地缓存（models--org--name 目录存在）。"""
    folder = "models--" + str(model_name).replace("/", "--")
    d = cache_dir() / folder
    if not d.is_dir():
        return False
    # 至少要有 snapshots 下的实际文件
    snaps = d / "snapshots"
    return snaps.is_dir() and any(snaps.iterdir())


def load_pretrained(loader, model_name: str, *args, quiet: bool = False,
                    prefer_cache: bool = True, **kwargs):
    """带离线回退的 from_pretrained。

    策略（这样断网时是**秒开**，而不是先卡 40 秒等超时）：
        1. 本地有缓存 -> 直接用缓存加载（不联网）
        2. 本地没有   -> 联网下载
        3. 联网失败但本地有缓存 -> 退回缓存
        4. 两边都没有 -> 抛出带修复指引的错误

    Args:
        loader:        例如 AutoTokenizer.from_pretrained
        model_name:    模型名或本地路径
        quiet:         离线回退时不打印提示
        prefer_cache:  本地有缓存时是否优先用缓存（默认 True）
    """
    # 本地路径 / 已显式指定离线：直接交给 loader
    if kwargs.get("local_files_only") or Path(str(model_name)).exists():
        return loader(model_name, *args, **kwargs)

    cached = is_cached(model_name)

    # ---- 1) 有缓存就先用缓存，避免每次都去联网握手
    if cached and prefer_cache:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return loader(model_name, *args, local_files_only=True, **kwargs)
        except Exception:
            pass          # 缓存坏了 -> 往下走联网

    # ---- 2) 联网加载
    try:
        return loader(model_name, *args, **kwargs)
    except Exception as online_exc:
        if not cached:
            raise RuntimeError(
                f"\n\n无法加载模型 '{model_name}'：\n"
                f"  * 联网失败：{type(online_exc).__name__}: {str(online_exc)[:160]}\n"
                f"  * 本地缓存：不存在（{cache_dir()}）\n\n"
                f"解决办法：\n"
                f"  1) 连上网络后重跑一次，模型会自动下载并缓存；\n"
                f"  2) 或设置国内镜像后再跑：\n"
                f'     $env:HF_ENDPOINT="https://hf-mirror.com"\n'
            ) from online_exc

        # ---- 3) 联网失败，退回缓存
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = loader(model_name, *args, local_files_only=True, **kwargs)
        except Exception:
            raise RuntimeError(
                f"\n\n无法加载模型 '{model_name}'：联网失败，且本地缓存不完整。\n"
                f"  * 联网错误：{type(online_exc).__name__}: {str(online_exc)[:160]}\n"
                f"  * 缓存目录：{cache_dir()}\n\n"
                f"解决办法：连上网络后重跑，或删除缓存目录让它重新下载。\n"
            ) from online_exc

        if not quiet and model_name not in _warned:
            _warned.add(model_name)
            print(f"[离线模式] 无法连接 HuggingFace，已改用本地缓存的 '{model_name}'"
                  f"（{cache_dir()}）")
        return obj


def describe_cache() -> str:
    """给自检脚本用：列出缓存里已有的模型。"""
    d = cache_dir()
    if not d.is_dir():
        return f"（缓存目录不存在：{d}）"
    models = sorted(p.name.replace("models--", "").replace("--", "/")
                    for p in d.iterdir() if p.is_dir() and p.name.startswith("models--"))
    if not models:
        return f"（缓存为空：{d}）"
    return f"{d}\n" + "\n".join(f"    - {m}" for m in models)
