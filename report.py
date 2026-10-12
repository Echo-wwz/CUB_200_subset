"""report.py —— 把 outputs/results/ 里的所有 json 汇总成一张对比表。

用法
----
    python report.py                 # 打印汇总表
    python report.py --save          # 同时写出 outputs/results/summary.csv 与 .md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from birdmm.utils import banner, section, to_builtin


def _result_files(pattern: str) -> list[Path]:
    """匹配结果 json，排除 *_errors.json 这类辅助文件（它们没有 test 字段）。"""
    return [f for f in sorted(config.RESULT_ROOT.glob(pattern))
            if not f.name.endswith("_errors.json")
            and not f.name.startswith(("summary", "04_ablation_summary"))]


def load_all() -> list[dict]:
    """读取所有结果 json，归一化成统一的记录。"""
    rows = []

    for f in _result_files("01_image_only_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append({
            "阶段": "阶段 2a 图像单模态",
            "模型": f"ResNet/{d.get('backbone')}" + ("(冻结)" if d.get("freeze_backbone") else "(微调)"),
            "Accuracy": d["test"].get("accuracy"),
            "Precision": d["test"].get("precision"),
            "Recall": d["test"].get("recall"),
            "F1": d["test"].get("f1"),
            "Top-5": d["test"].get("top5_accuracy"),
            "训练轮数": d.get("epochs"),
            "耗时(s)": d.get("train_time_sec"),
            "_file": f.name,
        })

    for f in _result_files("02_text_only_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append({
            "阶段": "阶段 2b 文本单模态",
            "模型": f"BERT({d.get('pooling')})" + ("(冻结)" if d.get("freeze_bert") else "(微调)"),
            "Accuracy": d["test"].get("accuracy"),
            "Precision": d["test"].get("precision"),
            "Recall": d["test"].get("recall"),
            "F1": d["test"].get("f1"),
            "Top-5": d["test"].get("top5_accuracy"),
            "训练轮数": d.get("epochs"),
            "耗时(s)": d.get("train_time_sec"),
            "_file": f.name,
        })

    for f in _result_files("04_fusion_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append({
            "阶段": "阶段 4 多模态融合",
            "模型": f"融合/{d.get('fusion_type')}",
            "Accuracy": d["test"].get("accuracy"),
            "Precision": d["test"].get("precision"),
            "Recall": d["test"].get("recall"),
            "F1": d["test"].get("f1"),
            "Top-5": d["test"].get("top5_accuracy"),
            "训练轮数": d.get("epochs"),
            "耗时(s)": d.get("train_time_sec"),
            "_file": f.name,
        })

    for f in _result_files("05_clip_zeroshot*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        for name, met in (d.get("zeroshot") or {}).items():
            rows.append({
                "阶段": "阶段 5 CLIP 零样本",
                "模型": f"CLIP {name}",
                "Accuracy": met.get("accuracy"),
                "Precision": None, "Recall": None, "F1": None,
                "Top-5": met.get("top5_accuracy"),
                "训练轮数": 0,
                "耗时(s)": None,
                "_file": f.name,
            })
        for direction, met in (d.get("retrieval") or {}).items():
            if not isinstance(met, dict):
                continue
            rows.append({
                "阶段": "阶段 5 CLIP 检索",
                "模型": direction,
                "Accuracy": None, "Precision": None, "Recall": None, "F1": None,
                "Top-5": met.get("R@5"),
                "训练轮数": 0, "耗时(s)": None,
                "_file": f.name,
            })
    return rows


def fmt(v, digits: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    p = argparse.ArgumentParser(description="汇总所有实验结果")
    p.add_argument("--save", action="store_true", help="写出 summary.csv 与 summary.md")
    args = p.parse_args()

    banner("实验结果汇总")
    rows = load_all()
    if not rows:
        print("还没有任何结果。先运行：python run_all.py")
        print(f"（结果目录：{config.RESULT_ROOT}）")
        return 1

    cols = ["阶段", "模型", "Accuracy", "Precision", "Recall", "F1", "Top-5", "训练轮数", "耗时(s)"]
    widths = [14, 30, 10, 11, 9, 9, 9, 9, 9]

    header = "".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "".join(fmt(r.get(c)).ljust(w) for c, w in zip(cols, widths))
        print(line)

    # 最佳模型
    accs = [(r, r["Accuracy"]) for r in rows if isinstance(r.get("Accuracy"), (int, float))]
    best = best_acc = None
    if accs:
        best, best_acc = max(accs, key=lambda x: x[1])
        section("最佳模型")
        print(f"{best['阶段']} / {best['模型']}  Accuracy = {best_acc:.4f}")

    # 多模态 vs 单模态
    img = [r for r in rows if r["阶段"] == "阶段 2a 图像单模态"]
    txt = [r for r in rows if r["阶段"] == "阶段 2b 文本单模态"]
    fus = [r for r in rows if r["阶段"] == "阶段 4 多模态融合"]
    if img and txt and fus:
        bi = max(img, key=lambda r: r["Accuracy"] or 0)
        bt = max(txt, key=lambda r: r["Accuracy"] or 0)
        bf = max(fus, key=lambda r: r["Accuracy"] or 0)
        section("单模态 vs 多模态")
        print(f"  最佳图像单模态 : {bi['模型']:<24} acc={bi['Accuracy']:.4f}")
        print(f"  最佳文本单模态 : {bt['模型']:<24} acc={bt['Accuracy']:.4f}")
        print(f"  最佳多模态融合 : {bf['模型']:<24} acc={bf['Accuracy']:.4f}")
        print(f"  融合相对最佳单模态的增益: {bf['Accuracy'] - max(bi['Accuracy'], bt['Accuracy']):+.4f}")

    if args.save:
        csv_path = config.RESULT_ROOT / "summary.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in cols})
        print(f"\nCSV -> {csv_path}")

        md_path = config.RESULT_ROOT / "summary.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 实验结果汇总\n\n")
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
            for r in rows:
                f.write("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n")
            if best is not None and best_acc is not None:
                f.write(f"\n**最佳模型**：{best['阶段']} / {best['模型']}，"
                        f"Accuracy = {best_acc:.4f}\n")
        print(f"Markdown -> {md_path}")

    print("\n图表在 outputs/figures/ 下：")
    figs = sorted(config.FIG_ROOT.glob("*.png"))
    for f in figs:
        print(f"  {f.name}")
    if not figs:
        print("  （还没有图，检查 matplotlib 是否安装）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
