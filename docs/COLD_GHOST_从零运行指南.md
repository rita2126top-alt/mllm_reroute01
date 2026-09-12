# MLLM-Reroute + Cold/Ghost：远程服务器 Conda 从零运行指南

完整正式训练、全部评测、全部诊断与可恢复总脚本请使用 [全量运行指南](COLD_GHOST_全量运行指南.md)。本页保留原安装与分步排错资料。

仓库：https://github.com/rita2126top-alt/mllm_reroute01

本文固定使用 **Linux x86_64、Bash、单张 NVIDIA GPU、Conda、Python 3.10**。代码交付时没有访问你的远程 GPU，也没有训练好的 Ghost 权重；权重由本文的两阶段训练流程生成。

原项目 81 个文件完整保留。新增功能位于 `cold_ghost/`、`scripts/cold_ghost/`、`environments/cold_ghost/`、`tests/cold_ghost/`。原 decision layer、attention 分数、Full top-k、预算和评测任务沿用原配置；改变的是 skip token 的状态维护。方法见 [实际实现说明](COLD_GHOST_METHOD.md)，检查结果见 [验证记录](COLD_GHOST_VALIDATION.md)。

## 1. 完整运行顺序与功能地图

按顺序执行：连接服务器 → 安装 Conda → 克隆代码 → 创建环境并安装依赖 → 下载 backbone → 原方法小样本验证 → 准备 GQA 和评测图片排除清单 → 训练流程小测试 → 完整训练 12 个权重 → 全部准确率评测 → 效率测试 → 消融与诊断。

| 功能 | scripts/cold_ghost/ 下的入口 | Ghost 权重 |
|---|---|---|
| 环境、GPU、评测框架检查 | doctor.py | 不需要 |
| 原文件完整性、语法、真实微型模型 CPU 测试 | check_project.py | 不需要 |
| 从全部原评测任务自动导出图片 hash | export_exclusions.py | 不需要，也不加载模型 |
| 固定 GQA 训练/验证划分 | prepare_data.py gqa | 不需要 |
| 单设置两阶段训练、续训 | train.py | 续训时需要部分权重 |
| 全设置串行训练、自动续训/跳过完成项 | train_matrix.py | 不需要 |
| 单设置原方法或 GC 准确率 | evaluate.py | GC 需要 |
| 准确率/profile/runtime 完整矩阵 | run_matrix.py | GC 需要 |
| 单设置 TFLOPs、KV、显存 | profile.py | GC 需要 |
| prefill 与端到端生成时间 | benchmark.py | GC 需要 |
| 过时状态、低秩残差、未来重激活诊断 | diagnose.py | 同设置 full、self 各一份 |

12 个 full checkpoint = **2 个 backbone × 2 套原 Reroute 决策方案 × 3 个预算档位**。compact 和 stagewise 共用同设置权重，所以覆盖 24 个 GC 推理设置。加上 38 个原设置，一共 62 个设置；每个设置的 `--tasks all` 包含 12 个任务条目。

本文每个代码块按从上到下执行。`--dry-run` 只列计划，不代表模型已经运行；`--limit` 和 `--smoke` 的结果不能当作正式全量实验。

## 2. 连接服务器并检查资源

以下第一行在自己电脑终端执行，替换用户名和 IP。

```bash
ssh YOUR_USER@YOUR_SERVER_IP
```

`ssh` 建立远程终端。此后所有命令在服务器执行。

```bash
uname -m
nvidia-smi
df -h "$HOME"
free -h
```

- `uname -m` 应输出 `x86_64`，与本文安装包架构一致。
- `nvidia-smi` 检查 GPU 和驱动。服务器驱动需支持 CUDA 12.8 wheel；最终以第 5 节实际 GPU 张量检查为准。
- `df -h "$HOME"` 检查用户目录磁盘剩余空间。两个 7B 模型、GQA 图片、完整评测集与缓存需要大容量存储，按数百 GB 级空间规划，以实际下载为准。
- `free -h` 检查内存。训练注释、图像排重和 dense teacher 状态会使用 CPU 内存。

冻结 backbone 并不意味着训练显存等于普通推理。训练需要通过学生路径反向传播，默认开启 decoder 激活检查点，batch size 1、梯度累积 16。交付时未测出真实 7B 的最低显存，先执行本文 smoke。Qwen 固定 BF16，服务器 GPU 应支持 BF16。

## 3. 安装 Conda

若已有能执行 `conda --version` 的 Conda，进入第 4 节。首次安装：

```bash
mkdir -p "$HOME/installers"
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME/installers/miniconda.sh"
bash "$HOME/installers/miniconda.sh" -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda --version
```

- `mkdir -p` 创建安装包目录。
- `wget -c` 下载并支持续传，`-O` 指定保存文件名。
- `bash ... -b -p ...` 将 Miniconda 无交互安装到自己的用户目录，不需要 root；不要覆盖已有安装。
- `source .../conda.sh` 在当前 Bash 会话启用 `conda activate`。
- `conda --version` 验证安装。

新开 SSH 会话后重新执行 `source`。安装方式依据 [Conda 官方 Linux 文档](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)。

## 4. 克隆仓库并创建专用环境

```bash
conda install -n base --override-channels -c conda-forge git -y
mkdir -p "$HOME/projects"
cd "$HOME/projects"
conda run -n base git clone https://github.com/rita2126top-alt/mllm_reroute01.git
cd mllm_reroute01
conda env create -f environments/cold_ghost/conda.yml
conda activate mllm_reroute_gc
python --version
which python
```

- 第一行从 conda-forge 给 base 环境安装 Git，避免依赖系统预装 Git。
- 第二、三行创建并进入项目父目录。
- `conda run -n base git clone` 使用刚安装在 base 中的 Git 下载完整仓库，无需依赖系统 Git 或当前是否激活 base。
- `cd mllm_reroute01` 进入根目录。后续除特别说明均在此执行。
- `conda env create` 按 YAML 创建 Python 3.10 环境，同时安装 Git、ffmpeg、patch、unzip、wget、tmux。
- `conda activate` 切换环境。
- 最后两行确认 Python 版本和解释器路径属于 `mllm_reroute_gc`。

以后重新登录：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mllm_reroute_gc
cd "$HOME/projects/mllm_reroute01"
```

三行分别加载 Conda、激活环境、回到仓库。

## 5. 安装依赖、评测框架和原补丁

```bash
bash scripts/cold_ghost/install_server.sh
```

这条命令依次检查系统和环境、安装原 requirements 与新增依赖、在仓库旁克隆 `lmms-eval-v0.7.1`、确认精确标签、按新增约束 editable 安装、调用原 grounding 补丁、执行 `pip check`、GPU/评测框架检查和 CPU 测试。任何步骤失败都会停止并返回非零。

核心版本固定为 torch 2.11.0+cu128、torchvision 0.26.0+cu128、transformers 5.4.0、Hydra 1.3.2、lmms-eval v0.7.1。新增 `constraints.txt` 还解决 Hydra 与数学解析依赖的 antlr 版本冲突。原 `requirements.txt` 完整保留。新增 FLOPs 入口使用 PyTorch 计数器，不要求安装 DeepSpeed 或编译额外 CUDA 扩展。

安装后可单独复查：

```bash
python -m pip check
python scripts/cold_ghost/doctor.py --require-gpu --require-eval --out experiments/cold_ghost/environment.json
python scripts/cold_ghost/check_project.py
```

- `pip check` 检查已安装包的声明依赖冲突。
- `doctor` 实际导入运行时、执行 CPU/GPU 张量、检查评测框架、12 个任务定义和原 grounding 补丁，保存 JSON；不会下载模型或数据。
- `check_project` 校验原 81 个文件、解析 Python/Bash、调用所有新增 CLI、检查实际导入并运行 CPU 测试。默认报告为 `experiments/cold_ghost/checks/report.json`。

`"ok": true` 才表示该检查通过。CPU 测试会隐藏 GPU 并关闭 Hugging Face 网络，使用随机初始化的微型 LLaVA/Qwen，不启动 7B 训练。

## 6. 设置缓存、设备并下载 backbone

统一将数据放在 `$HOME/mllm_data`；若用户目录空间不足，把第一行换成有写权限的大容量目录。

```bash
export MLLM_DATA_ROOT="$HOME/mllm_data"
mkdir -p "$MLLM_DATA_ROOT"
export HF_HOME="$MLLM_DATA_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export CONDA_ENV=mllm_reroute_gc
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" experiments/cold_ghost checkpoints/cold_ghost data/cold_ghost
hf download llava-hf/llava-1.5-7b-hf
hf download Qwen/Qwen2.5-VL-7B-Instruct
```

- `MLLM_DATA_ROOT` 是本文数据根变量；下一行创建它。
- `HF_HOME` 指定模型缓存，`HF_DATASETS_CACHE` 指定数据集缓存。
- `CUDA_VISIBLE_DEVICES=0` 只暴露服务器 0 号 GPU；换卡时统一修改这一行。
- `TOKENIZERS_PARALLELISM=false` 关闭 tokenizer 额外线程提示。
- `CONDA_ENV` 使原 shell 脚本调用新环境。
- `mkdir -p` 创建数据、权重、结果目录。
- 两条 `hf download` 下载原项目指定的模型。原配置中的模型 ID 保持不变，后续自动读取同一缓存。

若 Hugging Face 要求认证，执行 `hf auth login`，按交互提示输入自己的 token，不要写入仓库。下载中断后重复相同命令，会复用已完成文件。

`export` 只对当前会话有效；新开 SSH/tmux 会话后要重新设置，尤其保持相同 `HF_HOME`，避免重复下载。

## 7. 原方法小样本评测

```bash
python scripts/cold_ghost/evaluate.py --config baseline/llava15 --original --tasks pope --limit 8 --out experiments/cold_ghost/smoke_original_llava
python scripts/cold_ghost/evaluate.py --config llava15/avg192/llava15_ours_vs_fastv_avg192 --original --tasks pope --limit 8 --out experiments/cold_ghost/smoke_original_reroute
python scripts/cold_ghost/evaluate.py --config baseline/qwen25vl --original --tasks pope --limit 8 --out experiments/cold_ghost/smoke_original_qwen
```

三行分别验证原 LLaVA baseline、原 LLaVA Reroute、原 Qwen baseline。`--original` 运行原方法，不安装 GC；`--tasks pope` 选择 POPE；`--limit 8` 做小样本流程检查。POPE 若被 harness 展开为多个子任务，limit 按子任务生效。首次执行会下载对应数据。

`--out` 必须是尚不存在的目录，结果写在 `results.json`，配置写在 `run.json`。再次运行改成例如 `smoke_original_llava_r2`，不覆盖之前结果。小样本输出不能当作正式 benchmark 分数。

## 8. 下载 GQA 官方训练图片和问题

```bash
export GQA_ROOT="$MLLM_DATA_ROOT/gqa"
mkdir -p "$GQA_ROOT"
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip -O "$GQA_ROOT/questions1.2.zip"
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -O "$GQA_ROOT/images.zip"
unzip -n "$GQA_ROOT/questions1.2.zip" -d "$GQA_ROOT"
unzip -n "$GQA_ROOT/images.zip" -d "$GQA_ROOT"
export GQA_QUESTIONS="$(find "$GQA_ROOT" -type f -name train_balanced_questions.json -print -quit)"
export GQA_IMAGES="$GQA_ROOT/images"
test -f "$GQA_QUESTIONS"
test -d "$GQA_IMAGES"
```

- 前两行定义并创建 GQA 目录。
- 两条 `wget -c` 下载官方问题和原始图片压缩包，支持续传。
- 两条 `unzip -n` 解压且不覆盖已有文件。
- `find ... -print -quit` 找到正式训练问题文件，保存为 `GQA_QUESTIONS`。
- `GQA_IMAGES` 必须指向直接包含 `imageID.jpg` 的目录。
- 两条 `test` 返回 0 且无输出表示路径存在；失败时先检查解压位置。

下载地址与图片命名依据 [GQA 官方下载页](https://cs.stanford.edu/people/dorarad/gqa/download.html)。本项目使用图片，不需要下载预提取空间/目标特征。训练只输入官方 train_balanced 的图像和问题，不输入答案。

## 9. 排除评测图片，生成固定 8192/512 划分

```bash
python scripts/cold_ghost/export_exclusions.py --out data/cold_ghost/exclusions.json --cache-dir data/cold_ghost/exclusion_cache
```

此命令通过固定 lmms-eval 的真实任务加载器遍历 POPE、GQA、MMBench、MME 和 8 个 grounding 条目的评测图片，并加入仓库自带 3 张效率测试图片。它不加载 7B、不生成答案，但会下载完整评测数据并读取图像。

图片按照解码、校正方向后的 RGB 内容 hash 排重；每 250 条文档保存进度。中断后重跑相同命令，检查数据身份后继续。某任务下载失败时先解决数据源问题，不能跳过该范围开始正式训练。该排重识别相同 RGB 内容，不能保证识别裁剪、重新压缩后的所有近重复图片。

```bash
python scripts/cold_ghost/prepare_data.py gqa --questions "$GQA_QUESTIONS" --images "$GQA_IMAGES" --exclusions data/cold_ghost/exclusions.json --out data/cold_ghost/gqa_8192_512.json
```

此命令为每张 GQA 训练图片选择数值最小 question ID 的问题，按 seed 42 对应的固定 hash 顺序取样，排除评测图片和重复内容，生成互斥的 8192 张训练图、512 张内部验证图。清单保存问题、图片相对路径和审计 hash；内部验证不作为正式 benchmark。

有效图片不足 8704 张、文件缺失、排除范围不完整或 hash 不符都会报错。正式训练固定 8192/512，不能为了绕过错误缩小数量；小测试使用下一节明确标记的 smoke。

## 10. 在服务器验证两阶段训练

```bash
python scripts/cold_ghost/prepare_data.py gqa --questions "$GQA_QUESTIONS" --images "$GQA_IMAGES" --exclusions data/cold_ghost/exclusions.json --out data/cold_ghost/gqa_smoke.json --train-count 32 --val-count 4 --smoke
python scripts/cold_ghost/train.py --config llava15/avg192/llava15_ours_vs_fastv_avg192 --manifest data/cold_ghost/gqa_smoke.json --images "$GQA_IMAGES" --out checkpoints/cold_ghost/smoke_llava.pt --smoke --steps-per-stage 1 --device cuda
python scripts/cold_ghost/train.py --config qwen25vl/avg192/qwen25vl_ours_vs_fastv_avg192 --manifest data/cold_ghost/gqa_smoke.json --images "$GQA_IMAGES" --out checkpoints/cold_ghost/smoke_qwen.pt --smoke --steps-per-stage 1 --device cuda
```

- 第一行另建 smoke 32/4 清单，不修改正式清单。
- 第二、三行分别测试两个 backbone，各跑 warmup 1 个 optimizer step、rollout 1 个 optimizer step，每 step 累积 16 个样本；每阶段结束验证 4 张图。
- `--device cuda` 使用当前可见 GPU；默认开启 decoder 激活检查点。
- `--steps-per-stage` 必须和 `--smoke` 一起使用，防止短训练伪装成完整权重。

输出 `complete=true, smoke=true` 表示小训练完成。正式推理会拒绝 smoke 权重。后面使用正式清单从头训练，不拿 smoke checkpoint 作为正式结果。

## 11. 完整训练 12 个 full checkpoint

```bash
python scripts/cold_ghost/train_matrix.py --model all --tier all --ablation full --manifest data/cold_ghost/gqa_8192_512.json --images "$GQA_IMAGES" --output-dir checkpoints/cold_ghost --dry-run
python scripts/cold_ghost/train_matrix.py --model all --tier all --ablation full --manifest data/cold_ghost/gqa_8192_512.json --images "$GQA_IMAGES" --output-dir checkpoints/cold_ghost --device cuda
```

第一行列出 12 项计划，不加载模型；第二行才执行。每项固定：冻结 backbone，warmup 1 epoch、rollout 1 epoch；AdamW 学习率 `1e-4`、累积 16、梯度裁剪 1、seed 42。8192 个训练样本对应每阶段 512 steps，共 1024 optimizer steps；每阶段结束遍历 512 张内部验证图片。每 50 steps 和阶段边界保存断点。

训练矩阵串行启动独立子进程。重跑同一命令时，校验训练清单与配置，跳过已完成权重，自动续训兼容的部分权重。某项失败会记录并继续，其后最终退出码为非零；汇总位于 `checkpoints/cold_ghost/train_matrix_summary.json`。

文件名示例：

```text
checkpoints/cold_ghost/llava15__ours_vs_fastv__avg192__full.pt
checkpoints/cold_ghost/llava15__ours_vs_pdrop__avg192__full.pt
checkpoints/cold_ghost/qwen25vl__ours_vs_fastv__avg128__full.pt
checkpoints/cold_ghost/qwen25vl__ours_vs_pdrop__avg64__full.pt
```

`.pt` 只保存 Ghost 参数、优化器、训练状态与来源信息，不重复保存 7B 权重。同名 `.train.jsonl` 记录损失。compact/stagewise 共用同设置权重；不同 backbone、决策方案、tier、ablation 不混用。

## 12. 单项训练、主动停止与续训

单独完整训练一项：

```bash
python scripts/cold_ghost/train.py --config llava15/avg128/llava15_ours_vs_pdrop_avg128 --manifest data/cold_ghost/gqa_8192_512.json --images "$GQA_IMAGES" --out checkpoints/cold_ghost/single_example/llava15__ours_vs_pdrop__avg128__full.pt --device cuda
```

此命令的训练设置等价于矩阵中的对应项，输出放在独立 `single_example` 目录，避免与第 11 节已生成的权重重名。重复运行这个示例时也必须显式续训或使用新输出路径。

下面独立演示保存和恢复部分训练，不干扰主矩阵：

```bash
python scripts/cold_ghost/train.py --config llava15/avg192/llava15_ours_vs_fastv_avg192 --manifest data/cold_ghost/gqa_8192_512.json --images "$GQA_IMAGES" --out checkpoints/cold_ghost/resume_demo/full.pt --max-steps 20 --device cuda
python scripts/cold_ghost/train.py --config llava15/avg192/llava15_ours_vs_fastv_avg192 --manifest data/cold_ghost/gqa_8192_512.json --images "$GQA_IMAGES" --out checkpoints/cold_ghost/resume_demo/full.pt --resume checkpoints/cold_ghost/resume_demo/full.pt --device cuda
```

第一行在累计 20 optimizer steps 后保存 `complete=false` 并停止。第二行恢复参数、优化器、阶段、样本游标和随机状态，继续剩余训练。`--max-steps` 是累计上限，不是“再训练 N 步”；跑完续训时去掉它。

异常断开后从最近保存的梯度累积边界继续，未保存的计算需重跑。manifest、模型、Ghost 参数、smoke 属性不兼容会拒绝恢复，不能靠重命名文件混用。

## 13. GC 单项准确率和 stagewise

先用完整权重小测评测链路：

```bash
python scripts/cold_ghost/evaluate.py --config llava15/avg192/llava15_ours_vs_fastv_avg192 --checkpoint checkpoints/cold_ghost/llava15__ours_vs_fastv__avg192__full.pt --tasks pope --limit 8 --out experiments/cold_ghost/smoke_gc_llava
```

虽然权重训练完整，但 `--limit 8` 仍只是小样本流程检查。正式评测去掉 limit：

```bash
python scripts/cold_ghost/evaluate.py --config llava15/avg192/llava15_ours_vs_fastv_avg192 --checkpoint checkpoints/cold_ghost/llava15__ours_vs_fastv__avg192__full.pt --tasks all --out experiments/cold_ghost/full_gc_llava_avg192
python scripts/cold_ghost/evaluate.py --config llava15/avg192/llava15_ours_vs_fastv_stagewise_avg192 --checkpoint checkpoints/cold_ghost/llava15__ours_vs_fastv__avg192__full.pt --tasks all --out experiments/cold_ghost/full_gc_llava_avg192_stagewise
```

第一行 compact，第二行 stagewise，使用同一权重。Ghost 改变 skip 状态后，后续沿用原评分公式可能选到不同 token，这是状态变化的结果；没有换掉原 Full top-k 决策规则。

| --tasks | 内容 |
|---|---|
| pope | POPE |
| vqa | GQA、MMBench English dev、MME |
| refcoco | RefCOCO val/testA/testB |
| grounding | RefCOCO、RefCOCO+ 各 val/testA/testB；RefCOCOg val/test，共 8 条 |
| paper_main | POPE + 8 个 grounding 条目 |
| ablation | GQA、MMBench English dev、RefCOCO testA/testB |
| all | POPE + vqa + grounding，共 12 条 |

`paper_main` 与原项目一致，不自动包括全部 VQA；完整覆盖用 `all`。可传逗号分隔的任务 ID，例如 `pope,gqa,mme`。实际任务列表写入 `run.json`。

## 14. 全部 62 个准确率设置

```bash
python scripts/cold_ghost/run_matrix.py --model all --tier all --arms all --checkpoint-dir checkpoints/cold_ghost --tasks all --out experiments/cold_ghost/accuracy_all --dry-run
python scripts/cold_ghost/run_matrix.py --model all --tier all --arms all --checkpoint-dir checkpoints/cold_ghost --tasks all --out experiments/cold_ghost/accuracy_all
```

第一行列计划；第二行实际运行 38 原设置 + 24 GC 设置，包括 baseline、FastV、PDrop、Reroute compact/stagewise 与 GC compact/stagewise。

只运行原设置或只运行 GC：

```bash
python scripts/cold_ghost/run_matrix.py --model all --tier all --arms original --tasks all --out experiments/cold_ghost/accuracy_original
python scripts/cold_ghost/run_matrix.py --model all --tier all --arms gc --checkpoint-dir checkpoints/cold_ghost --tasks all --out experiments/cold_ghost/accuracy_gc
```

第一行 38 原设置无需 Ghost 权重；第二行 24 GC 设置。两行是同一完整矩阵的分组入口，不需要在已经完成 `accuracy_all` 后重复执行。

`--model` 可指定 `llava15` 或 `qwen25vl`；`--tier` 可指定 `avg192`、`avg128`、`avg64`。avg 数字是原平均 token 预算命名，并非每层 Full token 数。

矩阵输出 `plan.json`、每项独立结果与日志、最终 `summary.json`。`failed=0` 才表示全部成功。有失败查看对应 `.log`/`failure.json`，修复后用单项入口和新输出目录重跑。准确率矩阵不会覆盖旧目录，也不自动续写原结果。

## 15. TFLOPs、KV cache 和显存

原 Reroute 与 GC 都使用同一个新计数入口：

```bash
python scripts/cold_ghost/profile.py --config llava15/avg192/llava15_ours_vs_fastv_avg192 --original --out experiments/cold_ghost/profile_original_llava
python scripts/cold_ghost/profile.py --config llava15/avg192/llava15_ours_vs_fastv_avg192 --checkpoint checkpoints/cold_ghost/llava15__ours_vs_fastv__avg192__full.pt --out experiments/cold_ghost/profile_gc_llava
python scripts/cold_ghost/run_matrix.py --mode profile --model all --tier all --arms all --checkpoint-dir checkpoints/cold_ghost --out experiments/cold_ghost/profile_all
```

前两行单设置对比，第三行完整矩阵。使用原 `bench_data/manifest.json` 三张图和原问题；默认 1 pass、0 warmup，与原 profile shell 默认一致。

主要输出：

- `model_registered_flops` / `model_registered_tflops`：PyTorch 已注册算子口径，1 MAC=2 FLOPs，包含矩阵乘、卷积、融合 SDPA QK/AV、原决策 QK 和 Ghost 矩阵运算。
- `registered_flops_by_operator`：实际覆盖的算子明细。
- `gc_matrix_flops_analytic`：Ghost 矩阵运算子项，已包含在总矩阵 FLOPs，**不要再次相加**。
- `kv_bytes`：prefill 后实际 KV 存储。
- `auxiliary_state_bytes_at_prefill_end`：跳过状态等辅助张量存储。
- `prefill_peak_allocated_bytes`：CUDA 分配峰值。

此口径不包含未注册的逐元素算术、归一化、激活、排序和内存搬运；attention 按 dense QK/AV 等价算量，即使有 causal mask。论文/报告应注明这些边界。原 profiler 保留，但新的 GC 对照应两边都用本节入口，避免不同计数口径混用。

Ghost 带来额外成本。原 Full 预算不变，不代表加上 Ghost 后仍与原档位完全等 FLOPs；最终以实测输出为准。

## 16. prefill 和端到端生成时间

```bash
python scripts/cold_ghost/benchmark.py --config llava15/avg192/llava15_ours_vs_fastv_avg192 --original --out experiments/cold_ghost/runtime_original_llava
python scripts/cold_ghost/benchmark.py --config llava15/avg192/llava15_ours_vs_fastv_avg192 --checkpoint checkpoints/cold_ghost/llava15__ours_vs_fastv__avg192__full.pt --out experiments/cold_ghost/runtime_gc_llava
python scripts/cold_ghost/run_matrix.py --mode benchmark --model all --tier all --arms all --checkpoint-dir checkpoints/cold_ghost --out experiments/cold_ghost/runtime_all
```

前两行单项，第三行全部设置。默认遵循原 runtime：3 张图、2 warmup passes、5 正式 passes、确定性生成 64 个新 token，`use_cache=true`。

`prefill_ms` 是 prefill 时间，`e2e_ms` 是生成端到端时间，`decode_ms_per_token_estimate` 是 `(e2e_ms - prefill_ms)/64` 的差分估计，不是逐 decode step 插桩。CPU 图片处理不计入模型 GPU 时间。

增加重复次数的明确命令：

```bash
python scripts/cold_ghost/benchmark.py --config qwen25vl/avg192/qwen25vl_ours_vs_fastv_stagewise_avg192 --checkpoint checkpoints/cold_ghost/qwen25vl__ours_vs_fastv__avg192__full.pt --n-passes 10 --n-warmup-passes 2 --n-decode-tokens 64 --out experiments/cold_ghost/runtime_qwen_stagewise_10passes
```

`--n-passes 10` 改为 10 个正式 passes，其他条件不变；报告时明确次数。计时期间同卡不要运行其他实验。硬件、torch/CUDA 版本会写入结果。

## 17. 五种固定消融与三类诊断

主方法固定 `full`；`self`、`context`、`score`、`no_fresh` 是验证组件的消融，不是主方案的二选一实现。定义见方法文档。

```bash
python scripts/cold_ghost/train_matrix.py --model all --tier all --ablation all --manifest data/cold_ghost/gqa_8192_512.json --images "$GQA_IMAGES" --output-dir checkpoints/cold_ghost --device cuda
```

此命令枚举五种 ablation 共 60 个 checkpoint；原先完成的 12 full 经校验后跳过，新增 48 个消融权重。每项独立初始化并运行同样两阶段，不在 full 权重上关开关冒充独立训练。

```bash
for ablation in self context score no_fresh; do
    for model in llava15 qwen25vl; do
        for tier in avg192 avg128 avg64; do
            for schedule in ours_vs_fastv ours_vs_pdrop; do
                config="$model/$tier/${model}_${schedule}_${tier}"
                checkpoint="checkpoints/cold_ghost/${model}__${schedule}__${tier}__${ablation}.pt"
                python scripts/cold_ghost/evaluate.py --config "$config" --checkpoint "$checkpoint" --ablation "$ablation" --tasks ablation --out "experiments/cold_ghost/ablation_${ablation}/${model}__${schedule}__${tier}" || exit 1
            done
        done
    done
done
```

`for` 依次枚举四种消融、两个模型、三个预算和两套 schedule，共 48 个 compact 设置；`config` 与 `checkpoint` 分别拼出原配置和对应独立训练权重，`--tasks ablation` 固定评测 GQA、MMBench、RefCOCO testA/testB。每个 `done` 结束相应循环，`|| exit 1` 在失败时停止当前 Bash 会话。输出按消融和配置分目录，已有结果时应换新目录。A0 原 Reroute 与 A5 full 的相同四项指标从第 14 节对应 compact 结果读取；stagewise 一致性和效率由 full 主矩阵验证。

诊断 LLaVA 的一个设置：

```bash
python scripts/cold_ghost/diagnose.py --config llava15/avg192/llava15_ours_vs_fastv_avg192 --manifest data/cold_ghost/gqa_8192_512.json --images "$GQA_IMAGES" --checkpoint checkpoints/cold_ghost/llava15__ours_vs_fastv__avg192__full.pt --self-checkpoint checkpoints/cold_ghost/llava15__ours_vs_fastv__avg192__self.pt --out experiments/cold_ghost/diagnostics_llava_avg192.json --device cuda
```

该命令实际运行 dense teacher、原 Reroute、full GC 和独立 self，输出：

1. 按连续未 Full 更新 age 分组的重激活状态 MSE、cosine error，并给出共有重激活事件的配对对比。
2. Dense 残差低秩重建与 full/self 实际预测误差。
3. 下一个 decision 重激活预测的 AP、Ghost 预算召回等统计。

无正样本或共有事件时保留空/不可计算项，不伪造改进。诊断使用内部 512 张验证图，不代替 benchmark。将配置及两个权重同时替换为其他设置，即可覆盖其余 11 项。

## 18. 原项目所有入口仍可运行

```bash
export CONDA_ENV=mllm_reroute_gc
bash scripts/run_setting.sh baseline/llava15 --tasks pope --limit 8 --gpu 0
bash scripts/run_paper_table.sh all --tier all --tasks paper_main
bash scripts/run_paper_table.sh all --tier all --tasks vqa
bash scripts/run_profile.sh all --tier all
bash scripts/run_runtime_bench.sh all --tier all
```

- `CONDA_ENV` 让原 shell 调用新环境，不修改原脚本。
- `run_setting.sh` 原单项 POPE 小测试。
- 第一条 `run_paper_table.sh` 跑原 38 设置的 POPE + 8 grounding。
- 第二条补齐原 38 设置的 GQA/MMBench/MME。
- `run_profile.sh` 是原 TFLOPs/KV 入口；本文环境未安装其可选 DeepSpeed，原程序会警告并将 TFLOPs 记为 `null`，仍记录 KV。需要有明确算子覆盖的 TFLOPs 时使用第 15 节已提供的新增入口。
- `run_runtime_bench.sh` 是原时间入口。

原输出在 `experiments/logs`、`experiments/profile`、`experiments/runtime`；新入口按本文写在 `experiments/cold_ghost`，并拒绝写入上述原结果目录。执行原入口不需要 Ghost 权重；GC 对比使用第 13–17 节。

## 19. 长任务、日志和归档

训练前创建持久终端：

```bash
tmux new -s mllm_gc
```

此命令建立 `mllm_gc` 会话。进入后重新激活 Conda、进入仓库、设置第 6/8 节变量，再执行训练。按 `Ctrl+b`，松开，再按 `d` 可离开显示而保持进程运行。重新 SSH 后：

```bash
tmux attach -t mllm_gc
```

这条命令回到该会话。不要同时在同一 GPU 启动第二份训练。

```bash
tail -f checkpoints/cold_ghost/llava15__ours_vs_fastv__avg192__full.train.jsonl
python -m json.tool experiments/cold_ghost/accuracy_all/summary.json
```

第一行实时显示损失日志，`Ctrl+c` 只退出查看；第二行格式化显示矩阵汇总。检查 `failed=0`。各任务原始指标保留在其 `results.json`；POPE 与 MME 等量纲不同，不能直接简单平均。

保存运行环境：

```bash
git rev-parse HEAD > experiments/cold_ghost/code_commit.txt
python -m pip freeze > experiments/cold_ghost/pip_freeze.txt
conda env export > experiments/cold_ghost/conda_environment.yml
nvidia-smi > experiments/cold_ghost/gpu_environment.txt
```

四行依次保存代码 commit、Python 包、Conda 环境、GPU/驱动。`>` 将输出写入文件。连同训练 manifest、排除清单、checkpoint、配置和结果归档；数据、权重和实验输出不会随源码提交 Git。

## 20. 常见错误

| 错误/现象 | 处理步骤 |
|---|---|
| conda 找不到 | 先 source 第 3 节 conda.sh，核对真实安装路径。 |
| Python 包找不到 | 检查 which python，激活专用环境，重跑安装脚本与 pip check。 |
| torch/torchvision 导入失败 | 在干净专用环境按约束安装，不混合复制不同 wheel 文件。 |
| GPU 不可用 | 同会话检查 nvidia-smi、CUDA_VISIBLE_DEVICES，运行 doctor --require-gpu；驱动由服务器管理员处理。 |
| CUDA/BF16 不支持 | 保留固定软件与精度，在满足条件的 GPU 上运行，以实际张量测试为准。 |
| CUDA out of memory | 清理同卡其他任务，保留默认激活检查点，先 smoke；仍失败则用更大显存 GPU。不要降低原 max_pixels/Full 预算后声称设置不变。 |
| 数据下载失败 | 根据具体报错处理网络/数据源认证，保留缓存并重跑；不能省略失败任务的图片排除。 |
| grounding 补丁或任务缺失 | 重跑安装脚本与 doctor --require-eval，检查 editable 指向旁边的 lmms-eval-v0.7.1。 |
| checkpoint 被拒绝 | 确认 complete=true、smoke=false，backbone/schedule/tier/ablation 匹配；继续正确训练。 |
| 输出目录已存在 | 评测换新目录；训练显式 --resume 或重跑训练矩阵。 |
| exclusion cache identity changed | 换新 cache-dir 完整导出，再生成新 manifest；不同 manifest 不拼接续训。 |
| Bash 出现 ^M | 在 Linux 重新从仓库克隆，避免 Windows 编辑器改写 shell 换行。 |

## 21. 交付验证与服务器验收边界

交付已完成原文件 SHA256、Python/Bash 语法、全部新增 CLI、实际依赖导入、真实随机微型 LLaVA/Qwen 的 forward、generate、KV、梯度、训练与 checkpoint 测试。详细数量与版本见 [验证记录](COLD_GHOST_VALIDATION.md)。

**没有运行你的远程 GPU、真实 7B 全训练或全量 benchmark，也没有声称实测性能提升。** 服务器验收按第 5 节环境检查、第 7/10/13 节 smoke、完整训练 complete=true，以及矩阵 failed=0 逐步完成。

仅检查语法的命令：

```bash
python scripts/cold_ghost/check_project.py --syntax-only --report experiments/cold_ghost/checks/syntax.json
```

该命令不会运行模型测试，成功不等于运行验证通过；交付验收应使用不带 `--syntax-only` 的完整检查命令。
