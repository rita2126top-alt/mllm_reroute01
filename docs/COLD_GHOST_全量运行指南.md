# MLLM-Reroute + Cold/Ghost：完整、全量实验运行指南

这份指南只给出正式训练、完整数据评测和完整实验脚本。按固定方案运行两个 backbone、两套原 Reroute schedule、三个预算档位以及全部五种消融。所有准确率任务使用完整评测划分；训练执行完整两阶段。

仓库：[rita2126top-alt/mllm_reroute01](https://github.com/rita2126top-alt/mllm_reroute01)

全量总入口：`scripts/cold_ghost/run_full.sh`。环境配置：`scripts/cold_ghost/full_env.sh`。正式数据准备：`scripts/cold_ghost/prepare_full_data.sh`。实验调度和恢复：`scripts/cold_ghost/run_full_suite.py`。

## 1. “全部实验”具体包含什么

| 阶段 | 完整范围 | 数量与协议 |
|---|---|---|
| prepare | 原 LLaVA、Qwen backbone；GQA 官方训练图像/问题；全部评测图片排除 | GQA 8192 train + 512 独立 validation |
| train | 2 backbone × 2 schedule × 3 tier × 5 ablation | **60 个独立 checkpoint**；每个 warmup 1 epoch + rollout 1 epoch |
| accuracy | 原 38 个设置 + full GC 24 个设置 | **62 设置 × 12 个完整任务 = 744 个设置/任务组合** |
| profile | 同样 38 原设置 + 24 full GC | **62 设置**；原 3 图 cohort，1 pass、0 warmup |
| benchmark | 同样 38 原设置 + 24 full GC | **62 设置**；原 3 图 cohort，5 passes、2 warmup、生成 64 tokens |
| ablation | self/context/score/no_fresh × 12 compact 设置 | **48 设置 × 原 4 个完整消融任务 = 192 个设置/任务组合** |
| diagnostics | 全部 12 个 compact 设置，配套 full 与 self checkpoint | **12 组诊断**，每组使用全部 512 张内部验证图片 |

共 60 次训练和 246 个实验设置执行，即 **306 个底层训练/实验执行项**。总调度器将 60 次训练交给一个串行训练矩阵，因此其计划中显示 **247 个调度项**。

profile/benchmark 使用 3 张图是原项目的完整效率协议；并非缩小了完整评测集。诊断使用完整的内部 512 张验证图；准确率则遍历各 benchmark 的完整正式划分。不要把这三种数据范围混淆。

固定模型：

- `llava15`：`llava-hf/llava-1.5-7b-hf`。
- `qwen25vl`：`Qwen/Qwen2.5-VL-7B-Instruct`。
- schedule：`ours_vs_fastv`、`ours_vs_pdrop`。
- tier：`avg192`、`avg128`、`avg64`。
- ablation：`full`、`self`、`context`、`score`、`no_fresh`。
- compact 与 stagewise 共用对应 checkpoint；后者不重复训练。

原始 81 个文件保持不变；本次只增加全量调度和数据脚本、完善新入口与文档。方法公式及固定参数见 [方法说明](COLD_GHOST_METHOD.md)。

## 2. 服务器要求

固定使用 Linux x86_64、Bash、Conda、Python 3.10、单张 NVIDIA GPU。软件为原项目 torch 2.11.0+cu128、torchvision 0.26.0+cu128、transformers 5.4.0 和 lmms-eval v0.7.1。

Qwen 使用原 BF16，LLaVA 使用原 FP16；Ghost 参数使用 FP32。服务器驱动需要能够运行 CUDA 12.8 wheel。冻结 backbone 仍需对学生路径反向传播，训练比普通推理需要更多显存；保留默认激活检查点和 batch size 1。数据、两个 7B 模型及完整任务缓存需要大容量磁盘，按数百 GB 级空间规划，以实际下载量为准。

先在自己的电脑连接服务器：

```bash
ssh YOUR_USER@YOUR_SERVER_IP
```

将占位符替换为服务器用户名、地址。之后的命令全部在服务器执行。

```bash
uname -m
nvidia-smi
df -h "$HOME"
free -h
```

四行依次检查 CPU 架构、GPU/驱动、磁盘空间和 CPU 内存。

## 3. 从零安装 Conda、克隆项目

首次尚未安装 Conda 时：

```bash
mkdir -p "$HOME/installers"
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME/installers/miniconda.sh"
bash "$HOME/installers/miniconda.sh" -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda install -n base --override-channels -c conda-forge git -y
mkdir -p "$HOME/projects"
cd "$HOME/projects"
conda run -n base git clone https://github.com/rita2126top-alt/mllm_reroute01.git
cd mllm_reroute01
conda env create -f environments/cold_ghost/conda.yml
conda activate mllm_reroute_gc
bash scripts/cold_ghost/install_server.sh
```

逐行作用：

1. 创建安装包目录。
2. 下载 Linux x86_64 Miniconda 安装包，`-c` 支持续传。
3. 无交互安装到自己的用户目录；不要覆盖已有 Conda。
4. 使当前 Bash 会话可以激活 Conda 环境。
5. 从 conda-forge 给 base 安装 Git。
6. 创建项目父目录。
7. 进入父目录。
8. 使用刚安装的 Git 克隆完整仓库。
9. 进入项目根目录。
10. 创建 `mllm_reroute_gc`，包含 Python 3.10、Git、ffmpeg、unzip、wget、tmux 等。
11. 激活专用环境。
12. 安装原模型依赖、新约束、原标签源码版 lmms-eval、三个原 grounding 补丁，并检查依赖、GPU 和代码。

已安装 Conda 的服务器从 `source .../conda.sh` 开始，使用实际安装路径。已经克隆过本仓库的服务器使用以下更新命令，不再次克隆：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mllm_reroute_gc
cd "$HOME/projects/mllm_reroute01"
git pull --ff-only origin main
bash scripts/cold_ghost/install_server.sh
```

这五行依次加载 Conda、激活环境、进入仓库、快进更新代码、核对固定依赖。原模型权重和数据缓存不会被 Git 更新删除。

安装脚本采用官方 `v0.7.1` 源码 editable 安装，避免该版本 PyPI wheel 缺少任务模板的问题。它也固定了与 Hydra 相容的 antlr/数学解析包组合，不需要改原 requirements。安装完成后无需另行安装 DeepSpeed。

安装基础参考：[Conda Linux 文档](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)。

## 4. 配置数据盘、GPU、实验目录和原 MMBench 评分接口

建议先进入持久终端：

```bash
tmux new -s mllm_full
```

该命令创建 `mllm_full` 会话，SSH 断开后实验仍可继续。进入后执行：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mllm_reroute_gc
cd "$HOME/projects/mllm_reroute01"
export GC_DATA_ROOT="$HOME/mllm_data"
export CUDA_VISIBLE_DEVICES=0
source scripts/cold_ghost/full_env.sh
export API_TYPE=openai
export OPENAI_API_URL=https://api.openai.com/v1/chat/completions
export MODEL_VERSION=gpt-4o-2024-11-20
read -r -s -p "OpenAI API key for the original MMBench judge: " OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
```

- 前三行加载环境并进入根目录。
- `GC_DATA_ROOT` 是数据盘路径；如用户目录空间不足，将这一行替换为有写权限的大容量数据目录。
- `CUDA_VISIBLE_DEVICES=0` 选择服务器 0 号卡；整个全量流程串行使用该卡。
- `source full_env.sh` 统一导出数据、模型缓存、checkpoint 和结果路径。
- `API_TYPE`、`OPENAI_API_URL`、`MODEL_VERSION` 保留原 MMBench judge 协议。URL 必须是完整 chat/completions 端点。
- `read -s` 交互读取自己的 API key，不把密钥写在命令文本里；输入不回显。
- `printf` 为终端补换行。
- `export OPENAI_API_KEY` 让评测子进程读取密钥。不要把密钥提交到 Git。

原 MMBench 在部分回答无法直接提取选项时会请求该 judge，可能产生 API 费用。凭据必须能访问原指定模型；仅凭环境变量非空不能证明服务可用。脚本预检不发送 API 请求，正式评测才按原任务逻辑调用。接口或解析发生最终失败时，全量入口将该项标记为失败并保留诊断信息，不把随机回退分数当成可靠完成结果。

其余任务、训练、效率和诊断不需要这个评分接口。单独执行 `prepare/train/profile/benchmark/diagnostics` 不要求该 key；执行 `all/accuracy/ablation` 时必须准备好。

默认路径：

| 变量 | 默认位置/作用 |
|---|---|
| GC_PROJECT_ROOT | 实际仓库根目录，由脚本定位 |
| GC_DATA_ROOT | `$HOME/mllm_data` |
| HF_HOME | `$GC_DATA_ROOT/huggingface`，模型缓存 |
| HF_DATASETS_CACHE | `$HF_HOME/datasets`，评测数据缓存 |
| GQA_ROOT | `$GC_DATA_ROOT/gqa` |
| GQA_IMAGES | `$GQA_ROOT/images` |
| GC_MANIFEST | `data/cold_ghost/gqa_8192_512.json` |
| GC_EXCLUSIONS | `data/cold_ghost/exclusions.json` |
| GC_EXCLUSION_CACHE | `data/cold_ghost/exclusion_cache` |
| GC_CHECKPOINT_DIR | `checkpoints/cold_ghost` |
| GC_RUN_ROOT | `experiments/cold_ghost/full_run` |

表中的相对路径均以仓库根为准。改变数据盘时在首次 source 之前设置 `GC_DATA_ROOT`。若当前 shell 已存在旧 `HF_HOME`、`GQA_ROOT` 等变量，full_env 会保留它们；使用新终端设置，或在 source 前显式同步所有需要改变的路径。不要在一轮实验中途变更路径和清单。

需要 Hugging Face 认证时，运行 `hf auth login` 并按交互提示认证；这与 MMBench judge 的 API key 是两回事。

## 5. 一条正式命令执行完整项目

完成上述部署和配置后，执行：

```bash
bash scripts/cold_ghost/run_full.sh all
```

该命令实际执行：

1. 检查原 MMBench judge 配置、GPU、两个模型工厂和全部评测任务。
2. 下载两个 backbone。
3. 续传下载并解压 GQA 官方问题和图片。
4. 从全部评测任务导出真实 RGB 图片排除清单。
5. 固定生成 8192/512 正式训练/验证清单。
6. 串行完成 60 个正式 checkpoint。
7. 运行完整 62 设置主准确率矩阵。
8. 运行完整 62 设置 FLOPs/KV/显存矩阵。
9. 运行完整 62 设置 runtime 矩阵。
10. 运行完整 48 设置非 full 消融。
11. 对全部 12 个 compact 设置运行三类机制诊断。

**这里没有样本数量限制、缩短 epoch 或缩短训练 step。** 第一次运行会进行大规模模型/数据下载和正式计算。

需要同时把启动终端输出归档时，使用这一条等价启动方式：

```bash
mkdir -p "$GC_PROJECT_ROOT/experiments/cold_ghost/launcher_logs"
set -o pipefail
bash scripts/cold_ghost/run_full.sh all 2>&1 | tee "$GC_PROJECT_ROOT/experiments/cold_ghost/launcher_logs/full_$(date +%Y%m%d_%H%M%S).log"
```

第一行创建启动日志目录，第二行使管道正确传递实验失败状态，第三行启动相同全量任务，并把标准输出/错误显示和保存到时间戳日志。无需在直接 `all` 已经完成之后再执行这一组。

总调度器还会单独保存每项日志、精确子命令、尝试编号和结果，因此无需通过复制屏幕保存指标。

## 6. 中断后继续整个全量项目

重新 SSH 后：

```bash
tmux attach -t mllm_full
```

这条命令进入原持久会话。若原进程仍在运行，让它继续；不要同时启动第二个同目录任务。如果原进程已经退出：

```bash
bash scripts/cold_ghost/run_full.sh all --resume
```

完整恢复规则：

- 模型和 GQA 下载复用缓存，图片解压不覆盖已有文件。
- 图片排除导出从内容验证过的进度缓存继续。
- 正式 manifest 同 hash 保持旧文件不变；新旧 hash 不同会停止并保留候选清单，不覆盖旧身份。
- 完整兼容 checkpoint 跳过；部分兼容 checkpoint 从最近已保存的累积边界续训。
- 已完成的实验同时核对配置、manifest、checkpoint 身份和产物 hash，才会跳过。
- 失败或被中断的实验使用新的 attempt 目录重跑，不删除之前日志和结果。
- 同阶段中一项失败后仍收集后续项的结果。训练矩阵失败时，`all` 阻止依赖它的后续实验；其他实验阶段失败会被记录，并继续互不依赖的实验项，最后统一返回非零。

主模型推理一般需要从失败的设置开始重跑，不能保证从该设置中间的一张图续算；训练则支持 optimizer step 边界恢复。评测任务内部的缓存行为仍遵循原 harness。

同一输出根目录分阶段执行也使用 `--resume`。不要以手工创建空 results.json 或修改 summary 的方式跳过失败检查。

## 7. 同一全量流程的逐阶段命令

下面是第 5 节 `all` 的完整拆解。需要逐阶段执行时按顺序运行；每行都是全量阶段：

```bash
bash scripts/cold_ghost/run_full.sh prepare
bash scripts/cold_ghost/run_full.sh train --resume
bash scripts/cold_ghost/run_full.sh accuracy --resume
bash scripts/cold_ghost/run_full.sh profile --resume
bash scripts/cold_ghost/run_full.sh benchmark --resume
bash scripts/cold_ghost/run_full.sh ablation --resume
bash scripts/cold_ghost/run_full.sh diagnostics --resume
```

| 命令 | 实际工作 |
|---|---|
| prepare | 所有 backbone、GQA、完整评测图片排除和正式 8192/512 清单 |
| train | 全部 60 个 checkpoint |
| accuracy | 原 38 + full GC 24，完整 12 任务 |
| profile | 全部 62 设置 FLOPs、KV、辅助状态、显存 |
| benchmark | 全部 62 设置 prefill、端到端时间、decode 估计 |
| ablation | 四个非 full 消融的全部 48 compact 设置，原四任务全样本 |
| diagnostics | 全部 12 设置及所有 512 张内部验证图 |

首次 `train --resume` 在输出目录尚不存在时正常创建；后续阶段会保留前面的运行记录。已经完成 `all` 后再执行同样的阶段恢复命令，会验证并复用相应结果。

逐阶段执行的根 `summary.json` 记录最近一次调用，各次汇总保存在 `invocations/`。全部阶段完成后执行 `bash scripts/cold_ghost/run_full.sh all --resume`，可统一复核全部 247 个调度项并得到全流程汇总，已验证的计算不会重复运行。

## 8. 正式数据准备脚本具体做了什么

```bash
bash scripts/cold_ghost/prepare_full_data.sh
```

这是 `prepare` 阶段的数据子脚本，使用当前 source 的相同环境；执行完整下载和清单准备。它调用的官方模型下载命令是：

```bash
hf download llava-hf/llava-1.5-7b-hf
hf download Qwen/Qwen2.5-VL-7B-Instruct
```

两行下载原配置的两个模型并进入统一 Hugging Face 缓存。数据地址为 GQA 官方 `questions1.2.zip` 和 `images.zip`，来源：[GQA 官方下载页](https://cs.stanford.edu/people/dorarad/gqa/download.html)。

脚本查找唯一的 `train_balanced_questions.json`。若找到多个文件，不会随意取第一个；先在当前终端设置确切路径，再重跑：

```bash
export GQA_QUESTIONS="$GQA_ROOT/train_balanced_questions.json"
bash scripts/cold_ghost/prepare_full_data.sh
```

第一行必须改成实际解压位置，例如包含额外 `questions1.2` 子目录时写入该子目录；第二行重新执行完整准备。

数据身份规则：每张训练图取数值最小 question ID 的问题；固定 seed/hash 顺序；排除全部正式评测图和原效率图的相同 RGB 内容；训练图之间再去重。训练只输入图像和问题，不输入答案。全部 512 张内部验证图不参与参数更新。

已有数据清单出现不同 hash 时，候选文件将保留供检查。若确实要建立新的独立数据实验，必须同步使用新的 manifest、checkpoint 和结果目录，不能把新数据拼进旧 checkpoint 的续训。

## 9. 正式训练的完整底层命令

全量训练由以下实际命令完成：

```bash
python scripts/cold_ghost/train_matrix.py \
    --model all \
    --tier all \
    --ablation all \
    --manifest "$GC_MANIFEST" \
    --images "$GQA_IMAGES" \
    --output-dir "$GC_CHECKPOINT_DIR" \
    --device cuda
```

- `--model all`：两个原 backbone。
- `--tier all`：三个预算。
- `--ablation all`：full/self/context/score/no_fresh 五种，合计 60 权重。
- `--manifest`、`--images`：固定正式清单及图像根。
- `--output-dir`：checkpoint 根目录。
- `--device cuda`：当前唯一可见 GPU。

每项 warmup 1 epoch + rollout 1 epoch，每阶段 8192 张训练图；累积 16 对应每阶段 512 optimizer steps、合计 1024。学习率 `1e-4`、clip 1、seed 42、batch size 1，冻结原模型，默认 decoder activation checkpointing。每阶段结束验证完整 512 张内部验证图。

checkpoint 命名：

```text
<model>__<schedule>__<tier>__<ablation>.pt
```

例如 `llava15__ours_vs_fastv__avg192__full.pt` 与 `qwen25vl__ours_vs_pdrop__avg64__self.pt`。每项附带 `.train.jsonl`，原 7B 权重不重复保存。中断时重跑完整训练矩阵即可验证完成项并续训部分项。

若第 5/7 节已经执行训练，不必重复本节底层命令；它用于展示总脚本实际执行内容和单独管理全部训练。

## 10. 完整准确率、效率、消融与诊断命令

直接调用新的可恢复调度器，等价于相应 shell 阶段：

```bash
python scripts/cold_ghost/run_full_suite.py --stage accuracy --manifest "$GC_MANIFEST" --images "$GQA_IMAGES" --checkpoint-dir "$GC_CHECKPOINT_DIR" --out "$GC_RUN_ROOT" --device cuda --resume
python scripts/cold_ghost/run_full_suite.py --stage profile --manifest "$GC_MANIFEST" --images "$GQA_IMAGES" --checkpoint-dir "$GC_CHECKPOINT_DIR" --out "$GC_RUN_ROOT" --device cuda --resume
python scripts/cold_ghost/run_full_suite.py --stage benchmark --manifest "$GC_MANIFEST" --images "$GQA_IMAGES" --checkpoint-dir "$GC_CHECKPOINT_DIR" --out "$GC_RUN_ROOT" --device cuda --resume
python scripts/cold_ghost/run_full_suite.py --stage ablation --manifest "$GC_MANIFEST" --images "$GQA_IMAGES" --checkpoint-dir "$GC_CHECKPOINT_DIR" --out "$GC_RUN_ROOT" --device cuda --resume
python scripts/cold_ghost/run_full_suite.py --stage diagnostics --manifest "$GC_MANIFEST" --images "$GQA_IMAGES" --checkpoint-dir "$GC_CHECKPOINT_DIR" --out "$GC_RUN_ROOT" --device cuda --resume
```

五行分别执行完整准确率、完整 profile、完整 runtime、完整消融、全部诊断。`--out` 相同使所有阶段统一归档；`--resume` 复用已验证完成项。无需自己改配置名或手动补剩余 11 组诊断。

12 个正式准确率任务 ID：

```text
pope
gqa
mmbench_en_dev
mme
refcoco_bbox_rec_val
refcoco_bbox_rec_testA
refcoco_bbox_rec_testB
refcoco+_bbox_rec_val
refcoco+_bbox_rec_testA
refcoco+_bbox_rec_testB
refcocog_bbox_rec_val
refcocog_bbox_rec_test
```

固定版本中这 12 个都是原子任务；MME 一个任务下含多类指标。`paper_main` 只有 POPE + 8 个 grounding，因此不能用它替代这里的全部 12 项。

48 项非 full 消融使用原固定四任务：GQA、MMBench English dev、RefCOCO testA、RefCOCO testB，全部评测样本。原 Reroute 与 full 的对应四项分数直接从主准确率结果提取，避免重复推理。stagewise 的 full 一致性和效率由主矩阵覆盖。

## 11. 效率和诊断结果怎样解释

所有 full 效率设置保留原 3 张图片、问题、processor、Full top-k 和精度；GC 仅改变 skip token 状态。原设置和 GC 均使用相同新计数器，便于一致比较。

- `model_registered_tflops`：已注册矩阵/卷积/融合 attention 运算的 FLOPs，1 MAC=2 FLOPs；包括原 decision QK 与 Ghost 矩阵运算。
- `gc_matrix_flops_analytic` 已包含在总矩阵 FLOPs 内，不再次相加。
- `kv_bytes` 是实际 cache 张量占用；辅助状态和 CUDA 峰值另外报告。
- FLOPs 不包含未注册的逐元素算术、排序、内存搬运等，发布结果时附带计数口径。
- runtime 默认 5 passes、2 warmup、生成 64 tokens；`decode_ms_per_token_estimate` 是端到端减 prefill 后除以 64 的差分估计，不是逐 token 独立计时。
- 全部 12 组诊断报告 skip age 分组的重激活误差、残差低秩重建/full-self 预测误差、未来重激活预测质量，使用完整内部验证集。

加入 Ghost 有额外成本，不能把原 avg192/128/64 标签宣称为加入 Ghost 后依然完全等 FLOPs。预算保持原样，成本与效果都以实际输出为准。

## 12. 输出、失败定位与实验完成标准

统一输出根为 `$GC_RUN_ROOT`。其下按阶段、配置和 attempt 编号保存精确命令、日志和结果。旧 attempt 永不因重试而删除。总计划 `plan.json` 和最近一次调用的 `summary.json` 位于输出根，各次调用的汇总保存在 `invocations/attempt-XXXX/summary.json`；checkpoint 位于 `$GC_CHECKPOINT_DIR`。

同一输出根有操作系统排他锁，第二个同时运行的调度器会被拒绝。退出或崩溃后系统释放锁；保留的 `.full-suite.lock` 文件不是残留活进程，不需要手工删除。

完成标准：

1. 60 个 checkpoint 全部完成、正式清单身份一致。
2. accuracy/profile/benchmark 三个阶段分别完成 62 项。
3. ablation 完成 48 项；diagnostics 完成 12 项。
4. 汇总中无 failed 项、无中断项，所有目标阶段结束；脚本退出码为 0。
5. 最终评分失败或实际随机回退的项不能算有效完成，即使存在结果文件；中间请求重试后成功的项仍按原评分正常接受。

查看训练日志：

```bash
tail -f "$GC_CHECKPOINT_DIR/llava15__ours_vs_fastv__avg192__full.train.jsonl"
```

`tail -f` 实时显示此权重的阶段、step 和损失；按 Ctrl+c 退出查看，不会停止另一个 tmux 窗口中的训练。

查看总调度器输出文件：

```bash
find "$GC_RUN_ROOT" -maxdepth 2 -type f -name '*.json' -print
```

此命令列出计划/阶段汇总等 JSON 路径；单项错误查看该配置最新 attempt 的日志。更换 API 凭据后可恢复重跑失败评分项；密钥本身不会进入计划或身份记录。

失败处理固定为解决实际原因后重新执行 `bash scripts/cold_ghost/run_full.sh all --resume`，不删除结果树，也不修改原评测分数。

## 13. 单独完整复现原项目

如果只需执行原项目而无需 GC 训练，原入口仍完整保留：

```bash
export CONDA_ENV=mllm_reroute_gc
bash scripts/run_paper_table.sh all --tier all --tasks paper_main
bash scripts/run_paper_table.sh all --tier all --tasks vqa
```

第一行使原 shell 使用新环境；第二行跑原 38 设置的 POPE + 8 grounding；第三行补齐原 38 设置的 GQA/MMBench/MME。三行中没有缩小样本设置。原脚本不具备本次全量总调度器的结果 hash 恢复和 judge 失败判定，正式统一结果优先使用第 5/7 节。

只运行全部原设置的统一效率计数：

```bash
python scripts/cold_ghost/run_matrix.py --mode profile --model all --tier all --arms original --out experiments/cold_ghost/original_profile_full
python scripts/cold_ghost/run_matrix.py --mode benchmark --model all --tier all --arms original --out experiments/cold_ghost/original_runtime_full
```

第一行全部原 38 设置的 profile，第二行全部原 38 设置的 runtime。输出目录首次必须不存在；恢复需求使用新 full suite。原 `run_profile.sh` 未安装可选 DeepSpeed 时 TFLOPs 可能为 null，因此完整对比使用这里有明确覆盖的 PyTorch 计数器。

原完整入口的执行不是 `all` 之后还必须重复的一组实验；`all` 已包含其 38 设置。

## 14. 归档环境与重新登录

保存实际运行版本：

```bash
mkdir -p "$GC_PROJECT_ROOT/experiments/cold_ghost/environment_archive"
git rev-parse HEAD > "$GC_PROJECT_ROOT/experiments/cold_ghost/environment_archive/code_commit.txt"
python -m pip freeze > "$GC_PROJECT_ROOT/experiments/cold_ghost/environment_archive/pip_freeze.txt"
conda env export > "$GC_PROJECT_ROOT/experiments/cold_ghost/environment_archive/conda_environment.yml"
nvidia-smi > "$GC_PROJECT_ROOT/experiments/cold_ghost/environment_archive/gpu_environment.txt"
```

五行分别创建归档目录、保存代码 commit、Python 包版本、Conda 环境和 GPU/驱动。不要把所有环境变量整体导出，因为其中可能包含 API key。训练清单、排除清单、checkpoint、结果、日志和上述版本文件一起保留。

重新登录且新开会话时，重新执行第 4 节配置；已有 tmux 会话可直接 attach，原环境仍在。改变机器或数据位置后，优先保持原 manifest 与 checkpoint 内容；不要静默接受数据 hash 变化。

## 15. 检查范围与已知边界

本次交付会在本地进行 Bash/Python 语法、真实命令计划、无网络脚本模拟、恢复/失败处理及已有微型模型 CPU 测试。实际验证记录见 [全量入口验证记录](COLD_GHOST_FULL_VALIDATION.md)。

没有替你登录远程 GPU、执行 60 个真实 7B 训练或 936 个准确率设置/任务组合；文档提供的是完整可执行流程，性能和耗时需由服务器实际运行产生。936 = 主准确率 744 + 非 full 消融 192，不包含内部诊断。

遇到显存不足，保留原图像分辨率、Full 预算与精度，使用满足需求的 GPU；遇到数据下载失败，解决网络/授权后复用缓存恢复；遇到 judge 模型不可访问，解决该原模型访问条件后重跑，不以另一个评分方法替代并继续宣称协议相同。
