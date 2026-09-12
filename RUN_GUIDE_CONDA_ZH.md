# Reroute 项目远程服务器 Conda 完整运行指南

本文面向第一次在 Linux 远程 GPU 服务器上运行本仓库的用户。命令默认在 **Ubuntu/Linux + NVIDIA GPU + Bash** 环境执行；Windows 本地只用于 SSH 登录或上传代码。

> 本项目是推理评测与效率分析代码，不包含训练脚本。完整功能包括：LLaVA-1.5-7B 与 Qwen2.5-VL-7B 的无路由 baseline、FastV、PDrop、Reroute、stagewise Reroute；POPE/GQA/MMBench/MME/RefCOCO 系列准确率评测；prefill TFLOPs 与 KV cache 分析；端到端 GPU 运行时测试。

## 0. 开始前需要什么

建议条件：

- 一台 x86_64 Linux 服务器，能够访问 GitHub、Hugging Face 和 Python/PyTorch 软件源。
- NVIDIA GPU。7B 模型以 FP16/BF16 加载，建议至少 16 GB 显存；显存越宽裕越稳妥。代码是单 GPU 设计，`batch_size` 固定为 1。
- NVIDIA 驱动能支持所安装的 PyTorch CUDA wheel。服务器不必预装完整 CUDA Toolkit；只有 DeepSpeed TFLOPs 分析需要 `nvcc`。
- 模型、数据集、环境和缓存需要较多磁盘空间，建议至少预留 80–150 GB。完整 benchmark 数据会由 `lmms-eval/datasets` 从 Hugging Face 自动下载，实际占用取决于已缓存内容。

先登录服务器：

```bash
ssh 用户名@服务器地址
```

- `ssh`：建立远程终端连接。
- `用户名@服务器地址`：替换为管理员提供的账户和 IP/域名。

检查 GPU、驱动和磁盘：

```bash
nvidia-smi
df -h
```

- `nvidia-smi`：确认服务器能识别 NVIDIA GPU，并查看显存、驱动版本和当前占用。
- `df -h`：以易读单位查看磁盘剩余空间。

如果 `nvidia-smi` 不存在或报错，应先请管理员安装/修复 NVIDIA 驱动；Conda 不能代替内核驱动。

## 1. 把项目放到服务器

### 方式 A：服务器直接克隆（推荐）

```bash
mkdir -p ~/projects
cd ~/projects
git clone <本仓库的 Git URL> mllm-reroute-main
cd mllm-reroute-main
```

- `mkdir -p ~/projects`：创建项目父目录；已存在时不会报错。
- `cd ~/projects`：进入父目录。
- `git clone ... mllm-reroute-main`：下载仓库并指定目录名。
- `cd mllm-reroute-main`：进入项目根目录。后续命令均应在这里执行。

### 方式 B：从本机上传现有目录

在**本机终端**运行：

```bash
scp -r /本机路径/mllm-reroute-main 用户名@服务器地址:~/projects/
```

- `scp -r`：通过 SSH 递归上传整个目录。
- 冒号后的路径是服务器目标路径。

然后回到服务器：

```bash
cd ~/projects/mllm-reroute-main
ls
```

`ls` 应看到 `README.md`、`requirements.txt`、`configs`、`models`、`scripts`、`profiler` 等。

## 2. 安装并初始化 Conda

若服务器已有 Conda，可跳到下一节。检查：

```bash
conda --version
```

如果管理员提供了 Miniconda，但 `conda` 尚未加入 shell，可按实际安装位置执行类似：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc
```

- 第一行：把 Conda 函数加载到当前 Bash。
- `conda init bash`：把初始化片段写入 Bash 配置。
- `source ~/.bashrc`：不用重新登录就让配置生效。

若完全没有 Conda，请从 Miniconda 官方安装脚本安装；共享服务器上也可让管理员提供统一安装。安装完成后重新登录，再确认 `conda --version`。

## 3. 建立隔离环境

在项目根目录执行：

```bash
conda create -n reroute_release python=3.10 pip git -y
conda activate reroute_release
python --version
which python
python -m pip install --upgrade pip setuptools wheel
```

- `conda create -n reroute_release ...`：建立名为 `reroute_release` 的独立环境，安装 Python 3.10、pip 和 Git；`-y` 自动确认。
- `conda activate reroute_release`：激活环境。提示符通常会出现 `(reroute_release)`。
- `python --version`：应显示 Python 3.10.x。
- `which python`：应指向 Conda 环境，而不是 `/usr/bin/python`。
- 最后一行：升级 Python 打包工具，减少构建依赖时的兼容问题。

脚本默认环境名就是 `reroute_release`。若你使用别的名字，运行脚本前设置：

```bash
export CONDA_ENV=你的环境名
```

## 4. 安装 lmms-eval v0.7.1 和项目依赖

`lmms-eval` 必须以 editable 方式从固定 tag 安装，因为本项目会给它的 RefCOCO 任务文件打补丁。建议把它克隆到项目的同级目录，避免污染本仓库：

```bash
cd ..
git clone --branch v0.7.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
python -m pip install -e ./lmms-eval
cd mllm-reroute-main
python -m pip install -r requirements.txt
```

- `cd ..`：进入本项目父目录。
- `git clone --branch v0.7.1 --depth 1 ...`：只克隆上游 `v0.7.1` tag 的浅历史，保证评测代码版本一致。
- `pip install -e`：editable 安装；Python 直接使用该源码目录，补丁可以生效。
- `cd mllm-reroute-main`：回到本仓库根目录。
- `pip install -r requirements.txt`：安装锁定的 PyTorch、Transformers、Hydra、图像和数据处理依赖。

仓库默认安装 `torch==2.11.0+cu128` 与 `torchvision==0.26.0+cu128`。这里的 `cu128` 表示 PyTorch wheel 自带 CUDA 12.8 用户态运行库，并不表示驱动版本也必须显示为 12.8，但驱动必须足够新。

如果安装时提示找不到该 wheel，先不要随意混装版本。可根据服务器驱动兼容性，把 `requirements.txt` 中两行 PyTorch 固定项和 `--extra-index-url` 换成 PyTorch 官方存在且互相匹配的 CUDA wheel。仓库注释列出了 cu121/cu124 思路，但具体可用版本应以 PyTorch 官方索引为准。其余依赖仍用原文件安装。

安装 RefCOCO 补丁：

```bash
bash lmms_eval_patches/apply.sh
python -c 'from lmms_eval.tasks import TaskManager; TaskManager(); print("lmms-eval OK")'
python -c 'from lmms_eval.tasks.refcoco.utils_rec import BBOX_COORD_FORMAT, REFCOCO_PROMPT_STYLE; print(BBOX_COORD_FORMAT, REFCOCO_PROMPT_STYLE)'
```

- `apply.sh`：定位当前环境中的 `lmms_eval`，给 refcoco、refcoco+、refcocog 的 `utils_rec.py` 覆盖三个补丁；可重复执行。
- 第一条 Python 验证：加载全部任务定义，输出 `lmms-eval OK` 即通过。
- 第二条 Python 验证：确认补丁新增的坐标格式和 prompt 配置可以导入。

## 5. 配置 Hugging Face 缓存与访问

模型和数据首次运行会自动下载。建议把缓存放到容量大的数据盘，而不是默认的家目录：

```bash
mkdir -p /你的大容量磁盘/hf_cache
export HF_HOME=/你的大容量磁盘/hf_cache
export HF_HUB_CACHE=/你的大容量磁盘/hf_cache/hub
```

- `HF_HOME`：Hugging Face 的总缓存根目录。
- `HF_HUB_CACHE`：模型与数据仓库文件缓存目录。
- 这些 `export` 只对当前终端有效；可把它们加入自己的 `~/.bashrc` 以长期生效。

若下载提示模型或数据需要授权，先在 Hugging Face 网页接受对应许可，再登录：

```bash
huggingface-cli login
```

按提示粘贴个人 read token。不要把 token 写进仓库、日志或命令历史。新版本命令若没有 `huggingface-cli`，可使用 `hf auth login`。

可选的国内/受限网络方案应由服务器管理员提供 HTTP(S) 代理或已下载缓存。不要用不可信镜像处理私有 token。

## 6. 安装后必须做的自检

```bash
python -c 'import torch; print("torch", torch.__version__); print("CUDA wheel", torch.version.cuda); print("CUDA available", torch.cuda.is_available()); print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
python -c 'import transformers, hydra, lmms_eval; print("transformers", transformers.__version__); print("imports OK")'
python -m compileall -q models scripts profiler
```

- 第一行：验证 PyTorch、CUDA runtime 和 GPU 是否连通；`CUDA available` 必须为 `True`。
- 第二行：验证核心库可以导入，并显示 Transformers 版本。
- `compileall`：只做 Python 语法检查，不加载 7B 模型。

可再查看显存是否空闲：

```bash
nvidia-smi
```

## 7. 第一次跑通：最小评测冒烟测试

先只跑 LLaVA baseline 的 1 个 POPE 样本：

```bash
bash scripts/run_setting.sh baseline/llava15 --tasks pope --limit 1 --gpu 0
```

每部分含义：

- `bash scripts/run_setting.sh`：调用单配置评测调度器。
- `baseline/llava15`：选择 `configs/experiment/baseline/llava15.yaml`，不启用 token routing。
- `--tasks pope`：只运行 POPE 对象幻觉评测。
- `--limit 1`：只取 1 个样本，适合验证安装；它不能代表真实精度。
- `--gpu 0`：令该进程只看到物理 GPU 0。

首次运行最慢，因为要下载约 7B 参数模型和 POPE 数据。成功标志是日志出现 `RESULTS`、`Results saved to .../results.json` 和 `DONE`。

输出位置：

```text
experiments/logs/baseline/llava15/pope.log
experiments/logs/baseline/llava15/pope/results.json
experiments/logs/baseline/llava15/pope/samples.json
experiments/eval_results.jsonl
```

- `.log`：完整控制台日志。
- `results.json`：当前配置和任务的指标摘要。
- `samples.json`：配置开启 `log_samples` 时保存逐样本回答。
- `eval_results.jsonl`：所有实验逐行追加的全局索引。

然后测试一个真正的 Reroute 配置：

```bash
bash scripts/run_setting.sh llava15/avg192/llava15_ours_vs_fastv_avg192 --tasks pope --limit 1 --gpu 0
```

若这两条都完成，baseline、模型加载、路由 patch、数据下载和评测主链路已经跑通。

## 8. 单个实验的所有任务选择方式

通用格式：

```bash
bash scripts/run_setting.sh <配置路径，不带.yaml> [--tasks 任务选择器] [--limit N] [--gpu ID] [--log-dir 目录]
```

任务选择器：

| 选择器 | 实际任务 | 用途 |
|---|---|---|
| `pope` | POPE | 默认、最快入门 |
| `vqa` | GQA + MMBench English dev + MME | 通用 VQA |
| `refcoco` | RefCOCO val/testA/testB | 3 个标准定位 split |
| `grounding` | RefCOCO/RefCOCO+/RefCOCOg 共 8 splits | 全部 grounding |
| `ablation` | GQA + MMBench + RefCOCO testA/testB | 精简消融集合 |
| `paper_main` | POPE + 全部 8 个 grounding split | 论文主表集合 |
| 自定义 CSV | 例如 `pope,mme,refcoco_testA` | 自选任务 |

示例：

```bash
bash scripts/run_setting.sh baseline/qwen25vl --tasks vqa --gpu 0
bash scripts/run_setting.sh qwen25vl/avg128/qwen25vl_ours_vs_pdrop_avg128 --tasks grounding --gpu 0
bash scripts/run_setting.sh baseline/llava15 --tasks pope,mme,refcoco_testA --limit 20 --gpu 0
bash scripts/run_setting.sh baseline/llava15 --tasks pope --log-dir /你的大容量磁盘/reroute_logs --gpu 0
```

- 第一条：完整评测 Qwen baseline 的三个 VQA benchmark。
- 第二条：评测 Qwen、avg128、Ours-vs-PDrop 的全部定位任务。
- 第三条：每个自选任务最多跑 20 个样本，适合较完整的 smoke test。
- 第四条：把日志根目录放到指定大盘。

注意：脚本对每个 task 的失败采取“记录 FAIL 后继续”，所以批量结束后必须搜索失败：

```bash
grep -R "FAIL" experiments/logs
```

## 9. 复现论文配置矩阵

配置由模型、平均视觉 token 档位和路由方案组成：

- 模型：`llava15`、`qwen25vl`。
- 档位：`avg192`、`avg128`、`avg64`；越小压缩越激进。
- 每档 6 种：`fastv_K3`、`pdrop_earlyL2`、`ours_vs_fastv`、`ours_vs_pdrop`、`ours_vs_fastv_stagewise`、`ours_vs_pdrop_stagewise`。
- 每个模型另有 1 个不压缩 baseline。

方法含义：

| 配置 | 路由器 | 动作 | 含义 |
|---|---|---|---|
| baseline | none | none | 保留全部视觉 token |
| fastv_K3 | FastV | physical_delete | 第 3 层单次评分，未选 token 永久删除 |
| pdrop_earlyL2 | PDrop | physical_delete | 从第 2 层开始多阶段单调级联删除 |
| ours_vs_fastv | PDrop schedule | compact_route | 与 FastV FLOPs 对齐；未选 token 走 residual bypass，后续可重选 |
| ours_vs_pdrop | PDrop schedule | compact_route | 与 PDrop FLOPs 对齐；可恢复路由 |
| 两种 `_stagewise` | 同对应 ours | compact_route_stagewise | 路由决策相同，stage 内保持紧凑序列，主要用于实际加速 |

先用 limit 验证一个模型的一档 7 个配置：

```bash
bash scripts/run_paper_table.sh llava15 --tier avg192 --tasks pope --limit 1 --gpu 0
```

正式跑完整 POPE：

```bash
bash scripts/run_paper_table.sh llava15 --tier avg192 --tasks pope --gpu 0
```

其他常用组合：

```bash
bash scripts/run_paper_table.sh qwen25vl --tier avg128 --tasks vqa --gpu 0
bash scripts/run_paper_table.sh all --tier avg192 --tasks paper_main --gpu 0
bash scripts/run_paper_table.sh all --tier all --tasks paper_main --gpu 0
```

- `all --tier avg192`：两个 baseline + 两个模型各 6 个路由配置，共 14 次配置评测。
- `all --tier all`：两个 baseline + `2×3×6` 个路由配置，共 **38** 个配置。仓库 `bench_data/README.md` 中“44 configs”是文档计数错误，调度脚本实际生成 38 个。
- 每个配置还会按任务选择器拆分运行；全量 38 配置 × 9 个 paper_main benchmark 非常耗时。建议用 `tmux` 或作业调度系统。

## 10. 后台稳定运行与观察日志

使用 tmux：

```bash
tmux new -s reroute
conda activate reroute_release
cd ~/projects/mllm-reroute-main
bash scripts/run_paper_table.sh llava15 --tier avg192 --tasks paper_main --gpu 0
```

- `tmux new -s reroute`：建立名为 reroute 的持久会话。
- 运行后按 `Ctrl-b`，再按 `d`，即可断开但保持任务运行。
- 重新进入：`tmux attach -t reroute`。

另一终端观察 GPU 与日志：

```bash
watch -n 2 nvidia-smi
tail -f experiments/logs/baseline/llava15/pope.log
```

- `watch -n 2`：每两秒刷新 GPU 状态。
- `tail -f`：持续显示日志新增内容；按 `Ctrl-c` 退出观察，不会终止另一个 tmux 中的任务。

## 11. 运行时基准（无需下载 benchmark 数据集）

仓库自带 3 张 COCO 图片，运行时基准会测 prefill、固定长度生成和 decode token/s，并把模型回答写入 JSON。

先测单一 baseline，成本最低：

```bash
python profiler/bench_runtime.py \
  --config-name experiment/baseline/llava15 \
  --n-passes 1 \
  --n-warmup-passes 0 \
  --n-decode-tokens 16 \
  --out experiments/runtime/llava15_baseline_smoke.json
```

- `--config-name`：`configs/` 下配置路径，不带 `.yaml`。
- `--n-passes 1`：对 3 个样本各正式计时一次。
- `--n-warmup-passes 0`：冒烟时跳过预热；正式比较不建议跳过。
- `--n-decode-tokens 16`：每次强制生成 16 token。
- `--out`：结果 JSON 文件。

正式批量比较：

```bash
bash scripts/run_runtime_bench.sh llava15 --tier avg192 --n-passes 5 --n-warmup-passes 2 --n-decode-tokens 64 --gpu 0
bash scripts/run_runtime_bench.sh qwen25vl --tier all --n-passes 10 --n-warmup-passes 3 --n-decode-tokens 128 --gpu 0
```

结果写到 `experiments/runtime/*.json`，其中 `aggregate` 包含 prefill、端到端耗时、每 token decode 耗时和 token/s 的 p10/p50/p90/mean/std。

为保证公平，同一组对比应使用同一 GPU、同样的 pass/warmup/decode 参数，并尽量让 GPU 无其他任务。

## 12. TFLOPs 与 KV cache 分析

KV cache 分析即使没有 DeepSpeed 也能运行；TFLOPs 字段需要 DeepSpeed。先安装可选依赖：

```bash
conda install -c nvidia cuda-nvcc=12.4 -y
python -m pip install deepspeed==0.14.5
nvcc --version
```

- `cuda-nvcc`：安装编译 CUDA 扩展所需的编译器；版本应与环境和驱动兼容。
- `deepspeed==0.14.5`：安装代码预期的 FLOPs profiler。
- `nvcc --version`：确认编译器可用。

注意：仓库默认 PyTorch 是 cu128，而 README 示例的 `cuda-nvcc=12.4` 可能在某些机器上造成工具链不一致。如果 DeepSpeed 编译失败，应让 nvcc、PyTorch CUDA wheel 与驱动形成兼容组合；只需要 KV cache 时可不装 DeepSpeed。

单配置 smoke test：

```bash
python profiler/measure_metrics.py \
  --config-name experiment/baseline/llava15 \
  --n-passes 1 \
  --n-warmup-passes 0 \
  --out experiments/profile/llava15_baseline_smoke.json
```

批量分析：

```bash
bash scripts/run_profile.sh llava15 --tier avg192 --n-passes 3 --n-warmup-passes 1 --gpu 0
bash scripts/run_profile.sh all --tier all --n-passes 3 --n-warmup-passes 1 --gpu 0
```

结果位于 `experiments/profile/*.json`。分析范围是 **prefill only**；`aggregate.prefill_tflops_mean` 为平均 TFLOPs，`prefill_kv_cache_tokens_mean` 与 `prefill_kv_cache_MB_mean` 是 prefill 后 KV cache 大小。若 DeepSpeed 不可用，程序会警告并令 TFLOPs 为 `null`，KV 指标仍有效。

## 13. 使用自己的 3 张或更多图片做效率测试

运行时与 profiler 读取 `bench_data/manifest.json`。可替换图片并编辑 manifest，每项格式为：

```json
{
  "sample_id": 0,
  "image_id": "my_image_0",
  "category": "custom",
  "question": "Describe this image in detail.",
  "image": "sample_00.jpg",
  "image_WxH": [640, 427]
}
```

- `image` 必须是相对 `bench_data/` 的真实文件名。
- `image_WxH` 顺序是宽、高，并应与图像实际尺寸一致。
- 不同图像尺寸会改变视觉 token 数量，因此与论文自带 cohort 的耗时不再可直接比较。

这只改变效率微基准，不会改变 `lmms-eval` benchmark 数据。

## 14. Hydra 配置与高级覆盖

核心入口最终调用：

```bash
python scripts/run_eval.py \
  --config-dir "$PWD/configs" \
  --config-name experiment/baseline/llava15 \
  eval=pope \
  +eval.limit=1 \
  hydra.run.dir=experiments/manual_test
```

- `--config-dir`：指定配置根目录。
- `--config-name`：选择实验配置。
- `eval=pope`：把实验默认 eval 配置切换为 `configs/eval/pope.yaml`。
- `+eval.limit=1`：Hydra 新增 limit 字段。
- `hydra.run.dir=...`：指定本次结果目录。

一般优先使用 shell 调度器，因为它会设置日志、按任务循环并调用正确的 Conda 环境。直接入口适合调试配置。

RefCOCO prompt/坐标格式默认按模型自动设置：LLaVA 使用 `normalized + full`，Qwen2.5-VL 使用 `pixel + nuwa`。仅做受控消融时才覆盖：

```bash
export BBOX_COORD_FORMAT=pixel
export REFCOCO_PROMPT_STYLE=nuwa
bash scripts/run_setting.sh baseline/qwen25vl --tasks refcoco --limit 10 --gpu 0
unset BBOX_COORD_FORMAT REFCOCO_PROMPT_STYLE
```

错误配对可能让 RefCOCO 精度接近 0，因此正常复现不要设置这两个变量。

## 15. 常见故障排查

### `CUDA available False`

确认任务实际在 GPU 节点、`nvidia-smi` 正常、没有安装 CPU-only torch，并重新安装匹配的 PyTorch CUDA wheel。

### CUDA out of memory

```bash
nvidia-smi
```

先结束自己遗留的占卡进程或换空闲 GPU，再用 `--gpu ID` 指定。该仓库 batch size 已是 1。不要期待 routing 消除模型权重本身的显存；7B 权重仍需常驻。LLaVA profiler 明确 `.to("cuda")`，不能自动 CPU offload。

### 下载超时、连接失败或磁盘满

检查 `HF_HOME` 所在磁盘、服务器网络/代理和 Hugging Face 登录状态。下载中断后通常可重新执行，Hugging Face 会从缓存续传。

### `No module named lmms_eval`

```bash
conda activate reroute_release
python -m pip show lmms-eval
```

确认 `pip show` 的 Python 环境与 `which python` 一致；否则重新在当前环境执行 editable 安装。

### RefCOCO patch 失败

确认上游目录确实是 v0.7.1：

```bash
cd ../lmms-eval
git describe --tags --always
cd ../mllm-reroute-main
bash lmms_eval_patches/apply.sh
```

若源码被其他操作修改，重新克隆一个干净 v0.7.1 目录并重新 `pip install -e`，再打补丁。

### 脚本报 `conda: command not found`

非交互 shell 中先执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

路径按实际 Conda 安装位置调整。

### MMBench 名称看起来不同

仓库选择器写 `mmbench`，对应配置内部真正传给 lmms-eval 的任务名是 `mmbench_en_dev`。这是正常映射。

### 日志提到 WandB

`scripts/run_eval.py` 顶部 docstring 已过时。当前实现只输出本地 `results.json`、可选 `samples.json` 和全局 `eval_results.jsonl`，不需要安装或登录 WandB。

## 16. 推荐的完整执行顺序

按以下顺序最容易定位问题：

```bash
# 1) 环境、CUDA 和 import 自检
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
python -c 'from lmms_eval.tasks import TaskManager; TaskManager(); print("OK")'

# 2) baseline 单样本
bash scripts/run_setting.sh baseline/llava15 --tasks pope --limit 1 --gpu 0

# 3) routing 单样本
bash scripts/run_setting.sh llava15/avg192/llava15_ours_vs_fastv_avg192 --tasks pope --limit 1 --gpu 0

# 4) 自带图片的运行时 smoke test
python profiler/bench_runtime.py --config-name experiment/baseline/llava15 --n-passes 1 --n-warmup-passes 0 --n-decode-tokens 16 --out experiments/runtime/smoke.json

# 5) 一档完整 POPE
bash scripts/run_paper_table.sh llava15 --tier avg192 --tasks pope --gpu 0

# 6) 需要的完整准确率任务
bash scripts/run_paper_table.sh llava15 --tier avg192 --tasks paper_main --gpu 0

# 7) 正式 runtime 与 profile
bash scripts/run_runtime_bench.sh llava15 --tier avg192 --gpu 0
bash scripts/run_profile.sh llava15 --tier avg192 --gpu 0
```

只有前一步成功后再扩大任务规模。完整 38 配置应最后运行，并建议记录服务器 GPU 型号、驱动、PyTorch/CUDA 版本、commit/tag、缓存数据版本和所有命令参数，以保证结果可复现。
