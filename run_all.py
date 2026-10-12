"""run_all.py —— 一键跑完全部实验（带 --dry-run 看计划）。

设计原则
--------
* `--dry-run` 先看要跑什么、预计多久，确认后再真跑
* 分阶段、带编号，每个任务单独落盘日志到 outputs/logs/
* **单个任务失败不中断**，最后给汇总表
* 已完成的阶段可以跳过（--skip / --only）

用法
----
    python run_all.py --dry-run              # 只看计划，不执行（先跑这个！）
    python run_all.py                        # 按顺序跑完全部
    python run_all.py --quick                # 快速版：每个阶段只跑 2 个 epoch
    python run_all.py --only 4 5             # 只跑第 4、5 阶段
    python run_all.py --skip 1 2             # 跳过第 1、2 阶段
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from birdmm.utils import format_time

# 本机已知装好依赖的解释器（按优先级）
_KNOWN_INTERPRETERS = [
    r"C:\Users\Echo\.conda\envs\dl\python.exe",
    r"D:\anaconda3\python.exe",
]


def _has(interpreter: str, *modules: str) -> bool:
    """用指定解释器检查依赖是否齐全。"""
    if not Path(interpreter).exists():
        return False
    code = ("import importlib.util,sys;"
            "sys.exit(0 if all(importlib.util.find_spec(m) for m in sys.argv[1:]) else 1)")
    try:
        return subprocess.run([interpreter, "-c", code, *modules],
                              capture_output=True, timeout=120).returncode == 0
    except Exception:
        return False


def pick_interpreter() -> tuple[str, str]:
    """挑一个真正能跑实验的解释器。

    返回 (解释器路径, 说明)。当前解释器够用就用当前；
    不够用时自动切换到本机已装好依赖的那个 —— 这样即使用户在 VS Code 里
    选错了 Python，run_all.py 也照样能跑起来。
    """
    needed = ("torch", "torchvision", "transformers", "numpy", "PIL")

    if _has(sys.executable, *needed):
        return sys.executable, "当前解释器（依赖齐全）"

    for cand in _KNOWN_INTERPRETERS:
        if Path(cand).resolve() == Path(sys.executable).resolve():
            continue
        if _has(cand, *needed):
            return cand, f"自动切换到已装好依赖的解释器（当前解释器缺少依赖）"

    return sys.executable, "当前解释器（依赖不全，运行会失败）"


PY, PY_NOTE = pick_interpreter()


def _has_annotations() -> bool:
    return config.ANNOTATIONS.exists()


def _result_exists(name: str) -> bool:
    return (config.RESULT_ROOT / name).exists()


# ------------------------------------------------------------------ 任务表
def build_tasks(args) -> list[dict]:
    """返回任务列表。每个任务：{id, name, cmd, est, note, done_check}"""
    quick = args.quick
    ep = 2 if quick else None          # None = 用脚本默认值

    def ep_arg(default: int) -> list[str]:
        return ["--epochs", str(ep if ep else default)]

    tasks = [
        {
            "id": 0,
            "name": "数据准备（6:2:2 划分 + annotations.json）",
            "cmd": [PY, "prepare_data.py"],
            "est": "约 10 秒（首次）",
            "note": "只需跑一次；已存在 annotations.json 时自动跳过",
            "done": _has_annotations,
        },
        {
            "id": 1,
            "name": "阶段 2a 图像单模态（ResNet18）",
            "cmd": [PY, "01_image_only.py", *ep_arg(config.IMG_EPOCHS)],
            "est": "GPU 约 3~5 分钟 / CPU 约 15~25 分钟",
            "note": "冻结主干，只训练分类头",
            "done": lambda: _result_exists("01_image_only_resnet18.json"),
        },
        {
            "id": 2,
            "name": "阶段 2b 文本单模态（BERT）",
            "cmd": [PY, "02_text_only.py", *ep_arg(config.TXT_EPOCHS)],
            "est": "GPU 约 6~10 分钟 / CPU 较慢",
            "note": "BERT 全量微调；想快就加 --freeze",
            "done": lambda: _result_exists("02_text_only_bert_mean.json"),
        },
        {
            "id": 3,
            "name": "阶段 3 模型拆解与特征分析",
            "cmd": [PY, "03_model_dissection.py"],
            "est": "约 1 分钟",
            "note": "形状推导验证 + 特征图 + tokenizer/embedding/池化演示",
            "done": lambda: _result_exists("03_model_dissection.json"),
        },
        {
            "id": 4,
            "name": "阶段 4 多模态融合 + 消融（四种融合）",
            "cmd": [PY, "04_multimodal_fusion.py", "--fusion", "all", *ep_arg(config.FUSION_EPOCHS)],
            "est": "GPU 约 15~25 分钟（四种融合依次跑）",
            "note": "本实验的核心；含模态缺失实验与注意力可视化",
            "done": lambda: _result_exists("04_ablation_summary.json"),
        },
        {
            "id": 5,
            "name": "阶段 5 CLIP 零样本 + 图文检索",
            "cmd": [PY, "05_clip_zeroshot.py"],
            "est": "GPU 约 3~6 分钟",
            "note": "全程不训练参数；含 R@1/R@5 与提示词对照",
            "done": lambda: _result_exists("05_clip_zeroshot.json"),
        },
    ]

    if args.only:
        tasks = [t for t in tasks if t["id"] in args.only]
    if args.skip:
        tasks = [t for t in tasks if t["id"] not in args.skip]
    return tasks


# ------------------------------------------------------------------ 主流程
def main() -> int:
    p = argparse.ArgumentParser(description="一键跑完鸟类多模态分类实验")
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    p.add_argument("--quick", action="store_true", help="快速版：每个阶段只跑 2 个 epoch")
    p.add_argument("--only", type=int, nargs="+", default=None, help="只跑指定编号的阶段")
    p.add_argument("--skip", type=int, nargs="+", default=None, help="跳过指定编号的阶段")
    p.add_argument("--force", action="store_true", help="已完成的阶段也重跑")
    args = p.parse_args()

    print("=" * 78)
    print("  鸟类多模态分类实验 —— 总控脚本")
    print("=" * 78)
    print(config.describe())
    print(f"启动解释器 : {sys.executable}")
    print(f"实际使用   : {PY}   （{PY_NOTE}）")
    print()

    if not _has(PY, "torch", "torchvision", "transformers", "numpy", "PIL"):
        print("=" * 78)
        print("  [错误] 找不到装好依赖的 Python 解释器，无法运行实验")
        print("=" * 78)
        print("  请先跑环境自检，按提示修复：")
        print(f'      & "{sys.executable}" env_check.py')
        print()
        print("  或者手动指定一个装好 torch 的解释器，例如：")
        for cand in _KNOWN_INTERPRETERS:
            print(f'      & "{cand}" run_all.py')
        return 1

    tasks = build_tasks(args)

    # ---------------------------------------------------------- 计划
    print("执行计划")
    print("-" * 78)
    print(f"{'#':<4}{'阶段':<46}{'预计耗时':<28}")
    print("-" * 78)
    for t in tasks:
        status = "已完成" if (t["done"]() and not args.force) else "待执行"
        print(f"{t['id']:<4}{t['name']:<46}{t['est']:<28}")
        print(f"    {'':<42}[{status}]  {t['note']}")
    print("-" * 78)
    print(f"共 {len(tasks)} 个阶段。日志将写入 {config.LOG_ROOT}")
    print(f"结果（json）写入 {config.RESULT_ROOT}")
    print(f"图表（png） 写入 {config.FIG_ROOT}")

    if args.dry_run:
        print("\n[dry-run] 未执行任何任务。去掉 --dry-run 即可开始。")
        return 0

    # ---------------------------------------------------------- 执行
    print("\n" + "=" * 78)
    print("  开始执行")
    print("=" * 78)

    summary = []
    total_start = time.time()

    for t in tasks:
        if t["done"]() and not args.force:
            print(f"\n[{t['id']}] 跳过（已完成）：{t['name']}")
            print(f"    如需重跑请加 --force，或删掉对应的 outputs/results/*.json")
            summary.append((t["id"], t["name"], "跳过", 0.0, "已存在结果"))
            continue

        print(f"\n{'=' * 78}")
        print(f"[{t['id']}] {t['name']}")
        print(f"    命令: {' '.join(t['cmd'][1:])}")
        print(f"{'=' * 78}")

        log_file = config.LOG_ROOT / f"run_all_stage{t['id']}.log"
        t0 = time.time()
        try:
            # 直接继承 stdio：输出实时可见，同时由各脚本自己写日志文件
            proc = subprocess.run(t["cmd"], cwd=config.PROJECT_ROOT)
            code = proc.returncode
        except Exception as exc:
            print(f"[错误] 启动失败: {exc}")
            code = -1
        elapsed = time.time() - t0

        if code == 0:
            status = "成功"
        else:
            status = f"失败(exit={code})"
            print(f"\n[警告] 阶段 {t['id']} 失败，继续执行后续阶段（单个失败不中断）")
        summary.append((t["id"], t["name"], status, elapsed, ""))

    # ---------------------------------------------------------- 汇总
    print("\n" + "=" * 78)
    print("  执行汇总")
    print("=" * 78)
    print(f"{'#':<4}{'阶段':<44}{'状态':<14}{'耗时':>10}")
    print("-" * 78)
    for tid, name, status, elapsed, _ in summary:
        print(f"{tid:<4}{name:<44}{status:<14}{format_time(elapsed) if elapsed else '-':>10}")
    print("-" * 78)
    print(f"总耗时: {format_time(time.time() - total_start)}")

    failed = [s for s in summary if s[2].startswith("失败")]
    if failed:
        print(f"\n有 {len(failed)} 个阶段失败，请查看上面的报错信息。")
        return 1

    print("\n全部完成！")
    print("跑完看什么：")
    print(f"  1. 结果对比表  : python report.py          （汇总所有 json 成一张表）")
    print(f"  2. 训练曲线    : {config.FIG_ROOT / 'train_curve_*.png'}")
    print(f"  3. 混淆矩阵    : {config.FIG_ROOT / 'confusion_matrix_*.png'}")
    print(f"  4. 融合消融图  : {config.FIG_ROOT / 'ablation_*.png'}")
    print(f"  5. 注意力热力图: {config.FIG_ROOT / 'attention_*.png'}")
    print(f"  6. CLIP 相似度 : {config.FIG_ROOT / 'clip_similarity_matrix.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
