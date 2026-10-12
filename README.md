# 鸟类多模态分类：图像-文本联合表示

用 CUB_200 鸟类数据集，完整走一遍**单模态 → 双模态 → 跨模态对齐 → 多模态融合 → 消融分析**的技术路线。

---

## 这个项目在做什么

同一只鸟，图像能告诉你它的颜色、形状、姿态，却给不出"这是什么鸟"这个语义；文本能告诉你它的类别名和属性，却看不到它具体站在哪、是什么姿态。**多模态的核心问题不是"图像 + 文本 = 类别"，而是"图像和文本该如何被放进同一个空间里"。**

这个项目用 5 个阶段把这件事拆开做一遍：

| 阶段 | 做什么 | 回答什么问题 |
|---|---|---|
| 1 | 数据准备与划分 | 实验的数据基础怎么固定下来 |
| 2 | 图像单模态 / 文本单模态 | 每个模态单独能做到多好 |
| 3 | 模型拆解与特征分析 | 图像和文本各自被编码成了什么 |
| 4 | 四种融合方式 + 消融 | 两个模态怎么结合才有效 |
| 5 | CLIP 零样本与检索 | 不训练分类器能不能分类 |

---

## 一、快速开始

### 1. 环境自检（建议第一步）

```powershell
python env_check.py
```

一次看清：解释器、依赖、CUDA、数据、模型缓存。

### 2. 环境要求

| 项目 | 版本 |
|---|---|
| Python | 3.10 ~ 3.12 |
| torch | ≥ 2.0 |
| torchvision | ≥ 0.15 |
| transformers | ≥ 4.35 |
| 显卡 | 可选，CUDA 可用则自动启用 |

安装依赖：

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# torch 请按 https://pytorch.org 的命令装与你的 CUDA 匹配的版本
```

### 3. 跑起来

```powershell
python run_all.py --dry-run     # 先看要跑什么、预计多久
python run_all.py               # 全部跑完（GPU 约 30~45 分钟）
python run_all.py --quick       # 快速版：每阶段 2 个 epoch，约 10 分钟
python report.py --save         # 汇总所有结果
```

也可以单独跑某一个阶段：

```powershell
python prepare_data.py
python 01_image_only.py
python 02_text_only.py
python 03_model_dissection.py
python 04_multimodal_fusion.py --fusion all
python 05_clip_zeroshot.py
```

### 4. VS Code

打开本文件夹，按 `Ctrl+Shift+D` 打开**运行和调试**，下拉框里已配好 13 个配置，按 `F5` 即可运行。
另有任务（`Ctrl+Shift+P` → `Tasks: Run Task`）：环境自检、类型检查、各阶段、总控。

---

## 二、项目结构

```
CUB_200_subset/
├── data/                              数据
│   ├── images/<类别>/*.jpg            5871 张
│   ├── images/<类别>/*.txt            与 jpg 同名，每份 10 条描述
│   ├── text/<类别>/*.txt              原始文本副本
│   └── annotations.json               划分与标注（跑 prepare_data.py 生成）
│
├── config.py                          ★ 全局配置：路径、超参、模型名
│
├── birdmm/                            共享代码包
│   ├── data.py                        三个 Dataset + 图像预处理
│   ├── image_encoder.py               ResNet / MobileNetV2 图像编码器
│   ├── text_encoder.py                BERT 文本编码器 + masked mean
│   ├── fusion.py                      ★ 四种融合模块（核心）
│   ├── engine.py                      通用训练 / 评估循环
│   ├── metrics.py                     Acc / P / R / F1 / Top-5 / 混淆矩阵
│   ├── viz.py                         所有可视化
│   ├── hub.py                         HuggingFace 加载（带离线回退）
│   └── utils.py                       种子、日志、计时、依赖检查
│
├── prepare_data.py                    阶段 1  数据划分 + annotations.json
├── 01_image_only.py                   阶段 2a 图像单模态
├── 02_text_only.py                    阶段 2b 文本单模态
├── 03_model_dissection.py             阶段 3  模型拆解与特征分析
├── 04_multimodal_fusion.py            阶段 4  四种融合 + 消融
├── 05_clip_zeroshot.py                阶段 5  CLIP 零样本 + 检索
├── env_check.py                       环境自检
├── run_all.py                         总控
├── report.py                          结果汇总
├── requirements.txt
└── outputs/                           所有产物
    ├── logs/                          运行日志
    ├── results/                       指标 json + summary.csv/md
    ├── figures/                       所有图
    ├── checkpoints/                   模型权重
    └── features/                      特征向量（.npz）
```

---

## 三、各阶段详解

### 阶段 1 · 数据准备（`prepare_data.py`）

**输入**：`data/images/<类别>/*.jpg` + `data/text/<类别>/*.txt`（同名，每份 10 条描述）

**做三件事**：

1. **按类别分层**做 6:2:2 划分 —— 在**每个类别内部**切分，保证 100 个类在三份里都出现。如果全局随机切，小类可能整类落进测试集，指标就没意义了。
2. 生成 `annotations.json`，把划分**冻结**下来：

```json
{
  "meta": {"num_classes": 100, "num_captions": 10, "split_ratio": [0.6, 0.2, 0.2]},
  "classes": ["001.Black_footed_Albatross", ...],
  "class_prompts": {"001.Black_footed_Albatross": "a photo of a black footed albatross"},
  "splits": {
    "train": [{"image_path": "images/.../xxx.jpg", "captions": [...], "label": 0}, ...],
    "val": [...], "test": [...]
  }
}
```

3. 把 txt 复制到 `images/<类别>/` 下，让 jpg 与同名 txt 放在一起。

**实际结果**：100 类，5864 张可用样本 → **train 3515 / val 1177 / test 1172**（59.9% / 20.1% / 20.0%）。
有 7 张图缺同名 txt，已剔除并在日志中报告。

> **为什么划分要冻结**：所有脚本读同一份 `annotations.json`，单模态与多模态的结果才可直接比较 —— 这是消融实验成立的前提。

```powershell
python prepare_data.py                 # 已存在则跳过
python prepare_data.py --force         # 重新划分
python prepare_data.py --dry-run       # 只看统计
python prepare_data.py --no-sync-text  # 不复制 txt
```

---

### 阶段 2a · 图像单模态（`01_image_only.py`）

**模型**：ResNet18 / ResNet34 / ResNet50 / MobileNetV2 / MobileNetV3，去掉最后的池化层与全连接层，接一个新的分类头。

**默认配置**：冻结主干 + 只训分类头。ImageNet 预训练特征已经很强，这样几分钟就能跑完一轮，适合快速迭代；想要更高精度用 `--no-freeze`。

**输出**：Accuracy / 宏平均 Precision / Recall / F1 / Top-5，训练曲线，混淆矩阵，以及错误分析（最易混淆的类别对 + 具体错例含置信度）。

```powershell
python 01_image_only.py                            # ResNet18，冻结主干
python 01_image_only.py --backbone mobilenet_v2    # 换 MobileNetV2
python 01_image_only.py --no-freeze --lr 1e-4      # 解冻做全量微调
```

---

### 阶段 2b · 文本单模态（`02_text_only.py`）

**模型**：

```
描述 → BERT tokenizer → input_ids
     → BERT → last_hidden_state [B, L, 768]
     → attention_mask 过滤 padding 后 mean pooling → [B, 768]
     → Linear(768, 100) → 类别
```

**几个关键设计**：

* **每张图有 10 条描述**，训练时随机取一条（等价于文本增强），验证/测试固定取第 0 条（否则指标会抖动，无法比较）。
* **mean pooling 必须过滤 `[PAD]`** —— 一句话只有 10 个真实 token 却补到 48 长度时，直接平均会把有意义的向量严重稀释。阶段 3 会用数值量化这件事。
* 支持 `--pooling cls` 换成取 `[CLS]`，并用 `--compare-pooling` 实测两种池化的差异。

```powershell
python 02_text_only.py                     # BERT + mean pooling（默认）
python 02_text_only.py --pooling cls       # 换 [CLS] 池化
python 02_text_only.py --freeze            # 冻结 BERT，只训分类头（快）
python 02_text_only.py --compare-pooling   # 训练后对比两种池化
```

---

### 阶段 3 · 模型拆解与特征分析（`03_model_dissection.py`）

这个阶段不训练，只把两个编码器**拆开看**，是整个项目里信息密度最高的一环。

**图像侧**

| 内容 | 说明 |
|---|---|
| 逐层形状追踪 | 打印 `[3,224,224]` → `[64,112,112]` → … → `[512,7,7]` 每一层的输出形状 |
| token 化推导 | 验证 `[B,512,7,7] → flatten(2) → [B,512,49] → transpose(1,2) → [B,49,512]`，并与 `reshape+permute` 写法**数值比对，确认完全相等** |
| 两种用法 | 全局池化 `[B,512]` 供 Early/Late/Gated 融合；49 个 token 供 Cross-Attention 当 Query |
| 特征图可视化 | 按通道方差挑出最活跃的通道，看它们在响应鸟的哪个部位 |

**文本侧**

| 内容 | 说明 |
|---|---|
| Tokenizer 演示 | `a small blue bird` → 逐 token 打印 id，以及 `[CLS]/[SEP]/[PAD]/[MASK]/[UNK]` 的特殊 id |
| 三层 embedding | 分别打印 Token / Position / Segment Embedding 的形状与范数，再打印相加结果 |
| 两种池化 | `[CLS]` 向量 vs masked mean，形状与范数对比 |
| **mask 必要性量化** | 不做 mask 时句向量范数缩水多少、与正确向量的余弦相似度是多少（实测约 0.76，即已经偏得很远） |

**跨模态**

| 内容 | 说明 |
|---|---|
| 空间不可比 | 图像 512 维 vs 文本 768 维，直接点积报错 |
| 随机投影对照 | 强行拉到同一维度后，配对与非配对的相似度差 ≈ 0，图文检索 0 命中 —— 说明两个独立训练的编码器之间没有任何对齐机制 |
| 特征空间可视化 | 两个模态的特征分布，直观看到它们各占一块 |

这一步的结论直接引出阶段 4（用融合层硬拼）和阶段 5（用对比学习对齐）。

```powershell
python 03_model_dissection.py
python 03_model_dissection.py --num-samples 12 --text "a small blue bird"
```

---

### 阶段 4 · 多模态融合 + 消融（`04_multimodal_fusion.py`）★ 核心

四种融合模块**接口完全一致**：

```python
forward(image_feat, text_feat, attention_mask) -> {"logits": ..., 其余按需}
```

因为接口统一，换融合方式只需改 `--fusion` 一个参数 —— 这是消融实验能做干净的前提。

| 融合 | 结构 | 特点 |
|---|---|---|
| **Early** | `[图像 d] ⊕ [文本 d] → MLP → 类别` | 最简单，参数少；但两模态只是被拼在一起，没有显式交互 |
| **Late** | 两个分类头各出概率，`w·p_text + (1-w)·p_img` | `w` **可学习**；任一模态缺失都能工作；可解释 |
| **Cross-Attention** | 图像 token 作 Query，文本 token 作 Key/Value | 显式建模跨模态对齐；可导出注意力权重看"模型在看哪个词" |
| **Gated** | `a = σ(W[img;txt])`，`a·img + (1-a)·txt` | 逐样本决定更信谁；对不可靠文本有天然抑制 |

两个编码器的输出维度不同（512 / 768），由融合模块内部的投影层统一到公共维度 `d`（默认 256），这样四种融合的参数量才可比。

**额外做的三件事**：

1. **模态缺失实验** —— 对 Late / Gated 把文本置空、图像置空各跑一遍，量化"融合到底带来多少增益"
2. **注意力可视化** —— 画出 7×7 网格（每个图像位置最关注的词）+ 每个 head 的对比
3. **门控分布** —— gate 直方图 + 每类平均 gate，看哪些类更依赖图像、哪些更依赖文本

最后自动读取阶段 2 的结果，出**单模态 vs 多模态**总对比表与柱状图。

```powershell
python 04_multimodal_fusion.py --fusion all         # 四种依次跑，出消融对比
python 04_multimodal_fusion.py --fusion attention   # 只跑跨模态注意力
python 04_multimodal_fusion.py --fusion all --no-freeze --no-freeze-text --epochs 15
```

---

### 阶段 5 · CLIP 零样本与检索（`05_clip_zeroshot.py`）

思路的转变：

```
阶段 4：  image + text ──► 融合 ──► 分类器 ──► 类别编号
阶段 5：  image ──► 图像向量 ┐
                             ├──► 余弦相似度最大者 = 匹配
          text  ──► 文本向量 ┘
```

**没有分类器了**。类别不是被"输出"的，而是通过"和类别描述的相似度"被比较出来的。**全程不训练任何参数**，一次前向出结果。

**三种能力全部验证**：

| 能力 | 验证方式 |
|---|---|
| 图文匹配 / 检索 | 图像→文本、文本→图像两个方向的 **R@1 / R@5 / R@10** |
| 零样本分类 | **零样本 Accuracy**，并对比四种文本侧写法 |
| 通用表示 | 换数据集不用重训（讨论题） |

**文本侧四种写法对照**（回答"提示词到底值多少分"）：

| 变体 | 内容 |
|---|---|
| A. 只写类别名 | `black footed albatross` |
| B. 单模板提示词 | `a photo of a black footed albatross` |
| C. 模板集成（10 条） | 10 个模板的文本向量平均（prompt ensembling） |
| D. 真实描述原型 | 用该类别所有训练描述的平均向量当"分类器" |

> **一个容易踩的坑**：CLIP 有自己的归一化参数（`mean=(0.4815,0.4578,0.4082)`），与 ImageNet 的 `(0.485,0.456,0.406)` **不同**。如果把已经用 ImageNet 归一化过的 tensor 再交给 CLIPProcessor，等于归一化两次，零样本准确率会从 0.61 掉到 0.03。本工程保留 PIL 原图交给 processor 处理，并加了断言防止误用。

```powershell
python 05_clip_zeroshot.py                  # 完整版
python 05_clip_zeroshot.py --limit-test 200 # 只测 200 张，快速看效果
python 05_clip_zeroshot.py --no-retrieval   # 跳过 R@K
```

---

## 四、实测结果

以下数字来自本机完整跑完一轮（8 epoch，RTX 5070 Laptop 12GB，端到端约 33 分钟）。
原始数据在 `outputs/results/`，可用 `python report.py --save` 重新汇总。

### 4.1 主结果

| 模型 | Accuracy | Precision | Recall | F1 | Top-5 | 参数量（可训练） |
|---|---|---|---|---|---|---|
| 图像单模态 ResNet18 | 0.7398 | 0.7496 | 0.7401 | 0.7368 | 0.9488 | 11.2M（51.3K） |
| 文本单模态 BERT | 0.2765 | 0.2667 | 0.2754 | 0.2437 | 0.6015 | 110M（全量微调） |
| 融合 **Early** | **0.7534** | **0.7705** | **0.7546** | **0.7556** | 0.9573 | 121M（643K） |
| 融合 Late | 0.7381 | 0.7456 | 0.7393 | 0.7362 | 0.9488 | — |
| 融合 Cross-Attention | 0.7346 | 0.7457 | 0.7358 | 0.7326 | **0.9582** | — |
| 融合 Gated | 0.7176 | 0.7306 | 0.7195 | 0.7163 | 0.9565 | — |
| **CLIP 零样本**（未训练任何参数） | 0.6109 | — | — | — | 0.9343 | 0 |

> 测试集 1172 张，100 类，随机基线 0.01。
>
> 融合模型的 121M 参数里，绝大部分是两个冻结的主干（ResNet18 11M + BERT 110M），
> 真正训练的只有 643K 的投影层 + 融合层 + 分类头。

### 4.2 单模态 vs 多模态：三个值得注意的现象

**① 文本单模态只有 0.2765，而 CLIP 零样本有 0.6109**

这是最反直觉的一条：**没训练过的 CLIP 比训练过的 BERT 分类器好一倍多**。

原因是任务性质不同：
* BERT 分类器学的是"从 100 个类别里选一个"，而 CUB 的描述是通用外观（`a small blue bird`），几乎不提类别名（全部 58640 条描述里只有 **26 条**、即 0.04% 提到类别名）。文本里的信息量本来就不足以区分 100 种鸟。
* CLIP 是在 4 亿图文对上做过对比学习的，它见过"这种描述 ↔ 这种长相"的对应关系，用的是预训练阶段积累的视觉-语言对齐知识，而不是从这 5864 个样本里现学。

**② 融合只比最好的单模态高 1.4 个百分点**

Early 融合 0.7534 vs 图像单模态 0.7398，增益 `+0.0136`。原因很清楚：**图像模态太强（0.74），文本模态太弱（0.28），弱模态能补的信息有限。**

这也解释了为什么四种融合方式差距很小（0.7176 ~ 0.7534）—— 当两个模态强弱悬殊时，融合结构的设计空间被压缩了。

**③ Gated 融合的 gate 均值 = 0.804**

gate→1 表示更信图像。模型自己学出来"这次基本该听图像的"，与上面②的结论完全一致。这是 Gated 融合"逐样本决定信谁"的可解释性体现。

Late 融合的可学习权重同样印证了这一点：**w = 0.3921**（w 是文本的权重），即模型自动收敛到「39% 文本 + 61% 图像」。两种融合方式从不同机制出发，得到了同一个结论 —— 这说明"图像更重要"是数据本身的性质，不是某个融合结构的偶然。

### 4.3 模态缺失实验（Late / Gated 支持）

| 输入 | Late | Gated |
|---|---|---|
| 只有图像 | 0.7235 | 0.7090 |
| 只有文本 | 0.0358 | 0.0188 |
| 双模态 | 0.7381 | 0.7176 |
| **融合增益** | **+0.0145** | **+0.0085** |

> ⚠️ **这里的"只有文本 0.036"不能和文本单模态的 0.2765 直接比较**：融合脚本里 BERT 是**冻结**的（用通用特征），而文本单模态脚本里 BERT 是**微调**的。冻结的 BERT 特征配上只训了几轮的融合层，文本分支基本没被激活 —— 融合层发现图像分支已经 0.72 了，就学会了忽略文本。这是**模态竞争**的典型表现，不是 bug。
>
> 想验证的话：`python 04_multimodal_fusion.py --fusion late --no-freeze-text`

### 4.4 模型拆解的关键数字

| 项目 | 实测值 |
|---|---|
| 特征图 → token 两种写法是否等价 | **完全相等** |
| 图像特征维度 / 视觉 token 数 | 512 / 49 |
| 不做 mask 的池化范数缩水 | **7.35%** |
| 不做 mask 与正确向量的余弦相似度 | **0.7629** |
| 未对齐时配对 vs 非配对的相似度差 | **−0.0074**（几乎为 0） |
| 未对齐时的图文检索命中 | **0 / 6** |

第 3、4 行是"mean pooling 必须过滤 `[PAD]`"的量化证据：48 长度里只有 10 个真实 token，不做 mask 会让句向量偏离到余弦相似度只剩 0.76。

第 5、6 行说明：两个各自独立训练的编码器，输出空间**完全没有对齐**，配对和非配对的相似度没有系统性差别，检索等于瞎猜 —— 这正是阶段 5 存在的理由。

### 4.5 错误分析：模型到底在哪儿犯错

图像单模态错了 305 / 1172 张。最易混淆的类别对**全部是同类近缘物种**：

| 真实类别 | 被误判为 | 次数 |
|---|---|---|
| Herring_Gull（银鸥） | Glaucous_winged_Gull（灰翅鸥） | 5 |
| Eared_Grebe（黑颈鸊鷉） | Horned_Grebe（角鸊鷉） | 5 |
| Long_tailed_Jaeger（长尾贼鸥） | Pomarine_Jaeger（中贼鸥） | 4 |
| Brandt_Cormorant（加州鸬鹚） | Pelagic_Cormorant（海鸬鹚） | 4 |
| Red_faced_Cormorant（红脸鸬鹚） | Pelagic_Cormorant（海鸬鹚） | 4 |

这些在野外连观鸟者都要靠叫声和分布才能分辨。准确率最低的类别是 `009.Brewer_Blackbird`（0.083）—— 拟八哥属的几个种外观极其接近。

多模态融合后混淆模式**基本没变**（最大混淆对仍是 Long_tailed_Jaeger → Pomarine_Jaeger），说明当前配置下文本没能提供新的判别信息，与 4.2 的结论一致。

### 4.6 CLIP 零样本与检索

**文本侧四种写法**（全程未训练参数）：

| 变体 | Accuracy | Top-5 |
|---|---|---|
| A. 只写类别名 | 0.5904 | 0.9019 |
| B. 单模板提示词 | 0.6007 | 0.9309 |
| C. 模板集成（10 条） | **0.6109** | **0.9343** |
| D. 真实描述原型 | 0.3166 | 0.7270 |

**这四行的排序本身就是结论**：提示词写得好不好，能带来 **+2 个百分点**（A→C）；而用数据集里的真实描述反而最差（D）—— 因为那些描述是"这只鸟长什么样"的属性描述，几乎不含类别名，把它们平均起来当分类器权重，等于丢掉了类别信息。

**检索 R@K**：

| 方向 | R@1 | R@5 | R@10 |
|---|---|---|---|
| 图像 → 文本 | 0.0247 | 0.0879 | 0.1442 |
| 文本 → 图像 | 0.0171 | 0.0572 | 0.1084 |

> **这两个数字看着很低，但需要正确解读**：gallery 是 1172 条**各自唯一**的描述，而描述内容是通用外观（"a small blue bird"）。同一种鸟的 10 张照片，描述几乎可以互换 —— 实例级配对本来就不唯一。实测 R@1 是随机基线（1/1172 ≈ 0.00085）的 **29 倍**，说明模型确实在工作，只是这个指标不适合衡量"外观描述 ↔ 单张照片"的匹配。
>
> 真正有意义的是**类别级检索**：用 100 个类别提示词当 gallery，Top-1 = **0.6109**，与零样本分类的 accuracy 完全一致（本来就是同一个数）。
>
> 想看对齐效果，看 `outputs/figures/clip_alignment_evidence.png`：配对相似度均值 0.292 vs 非配对 0.261，配对明显更高 —— 对比阶段 3 未对齐时的"差值为 −0.007"，这才是 CLIP 的价值所在。

---

## 五、跑完看什么

| 想看什么 | 去哪看 |
|---|---|
| 所有指标汇总成一张表 | `python report.py --save` → `outputs/results/summary.md` |
| 训练曲线（loss / acc） | `outputs/figures/train_curve_*.png` |
| 混淆矩阵 | `outputs/figures/confusion_matrix_*.png` |
| 四种融合方式对比 | `outputs/figures/ablation_fusion.png` |
| 单模态 vs 多模态 | `outputs/figures/ablation_all_models.png` |
| 跨模态注意力（模型在看哪个词） | `outputs/figures/attention_attention.png` |
| 门控分布 | `outputs/figures/gate_distribution_gated.png` |
| ResNet 空间特征图 | `outputs/figures/feature_map_resnet.png` |
| CLIP 图文相似度矩阵 | `outputs/figures/clip_similarity_matrix.png` |
| CLIP 对齐证据 | `outputs/figures/clip_alignment_evidence.png` |
| 每个脚本的完整输出 | `outputs/logs/*.log` |

---

## 六、配置说明

全部超参集中在 `config.py`，命令行参数会覆盖它。

| 分组 | 关键项 | 默认值 |
|---|---|---|
| 数据 | `SPLIT_RATIO` / `SEED` / `NUM_CAPTIONS` | `(0.6,0.2,0.2)` / `42` / `10` |
| 图像 | `IMG_BACKBONE` / `IMG_BATCH_SIZE` / `IMG_EPOCHS` / `IMG_FREEZE` | `resnet18` / `32` / `8` / `True` |
| 文本 | `BERT_NAME` / `TXT_BATCH_SIZE` / `TXT_EPOCHS` / `TXT_POOLING` | `bert-base-uncased` / `32` / `8` / `mean` |
| 融合 | `FUSION_DIM` / `FUSION_BATCH_SIZE` / `FUSION_EPOCHS` / `FUSION_FREEZE` | `256` / `32` / `8` / `True` |
| 运行 | `NUM_WORKERS` | `0`（Windows 下多进程易出问题） |

---

## 七、常见问题

**Q：跑不动 / 显存不够？**

调小 `config.py` 里的 `IMG_BATCH_SIZE` / `TXT_BATCH_SIZE` / `FUSION_BATCH_SIZE`（如 8），或用 `--limit-batches` 限制每轮 batch 数。CPU 也能跑，只是慢。

**Q：结果不理想？**

这是 **100 类细粒度鸟类分类**，任务本身很难。默认配置是"冻结主干 + 只训融合层"，图的是快。想要更高精度：

```powershell
python 04_multimodal_fusion.py --fusion attention --no-freeze --no-freeze-text --epochs 15 --lr 5e-5
```

另外 `--limit-batches` 只是冒烟测试用的，**正式结果不要加这个参数**。

**Q：下载模型卡住 / 报 `httpx.ConnectTimeout`？**

工程做了**离线回退**：加载顺序是「先查本地缓存 → 没有再联网 → 联网失败退回缓存」。只要模型下载过一次，之后断网也能跑，而且是秒开（不会卡在超时等待上）。

首次下载时如果网络不通，用镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"   # 当前终端生效
```

模型缓存在 `~/.cache/huggingface`，`python env_check.py` 会告诉你哪些已缓存好。

**Q：报错 `No module named 'torch'`？**

用错 Python 解释了。先跑 `python env_check.py`，它会直接告诉你该怎么办。

如果本机有多个解释器，可以用 `run_all.py` —— 它检测到当前解释器缺依赖会**自动切换**到能用的那个。

**Q：换别的数据集？**

改 `config.py` 里的 `DATA_ROOT` / `NUM_CLASSES`，数据摆成 `images/<类别>/*.jpg` + 同名 `.txt`（每份 10 行描述），然后：

```powershell
python prepare_data.py --force
```

**Q：想调划分比例？**

改 `config.py` 的 `SPLIT_RATIO`，然后 `python prepare_data.py --force`。注意重新划分后各阶段的结果不再可比，建议全部重跑。

---

## 八、复现说明

* 所有脚本用 `config.SEED = 42` 固定随机种子（Python / numpy / torch / cudnn 全部固定）
* 6:2:2 划分冻结在 `annotations.json`，不重新划分就不会变
* 每个脚本的结果 json 里都记录了完整超参，可直接对照复现
* 训练时验证集选最佳 epoch，测试集只用于最终报告（不参与调参）

---

## 九、依赖

```
必需：torch / torchvision / transformers / pillow / numpy
可选：tqdm（进度条）/ PyYAML / matplotlib（出图）/ scikit-learn（指标）
```

可选项缺失时脚本会自动降级（比如没装 sklearn 就用内置的 numpy 实现算指标，没装 matplotlib 就跳过绘图），不会报错中断。
