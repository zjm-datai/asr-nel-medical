# SpeechSearcher GPU 服务器训练说明

本文用于当前 200 条整句、101 个候选实体表面形式的 SS 先导实验。该实验用于
验证检索方案，不是最终生产模型。

## 1. 需要上传的内容

上传工作区根目录中的 `asr-nec-model/`、`data/` 和 `weights/`。目录结构应为：

```text
workspace/
├── asr-nec-model/       # 只放代码、测试和文档
├── data/                # 音频、清单、ASR 和特征缓存
└── weights/base.pt      # Whisper base 权重
```

## 2. 创建环境

以下命令在代码仓库目录运行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv sync
```

然后安装与服务器 CUDA/驱动匹配的 PyTorch。下面是 CUDA 12.8 示例；如果服务器
环境不同，应按照 PyTorch 官方安装选择器替换 `cu128`：

```bash
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

最后一条必须输出 `True` 和实际 GPU 名称。若输出 `False`，不要开始训练。

## 3. 提取冻结 Whisper 特征

只需执行一次：

```bash
uv run python scripts/extract_ss_features.py \
  --model ../weights/base.pt \
  --device cuda \
  --batch-size 16
```

输出位于：

```text
../data/speech_searcher/ss_features/
```

脚本支持断点续跑。中断后执行同一条命令即可，不会重新计算完整文件。

建议 batch size：

| GPU 显存 | 特征提取 batch size |
| --- | ---: |
| 8 GB | 4–8 |
| 12–16 GB | 8–16 |
| 24 GB 及以上 | 16–32 |

如果出现 CUDA OOM，将 batch size 减半后重跑。

## 4. 训练 SS

推荐的先导训练命令：

```bash
uv run python scripts/train_speech_searcher_pilot.py \
  --device cuda \
  --epochs 20 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --learning-rate 1e-4 \
  --num-workers 4 \
  --patience 5 \
  --output-dir ../runs/ss_pilot
```

训练使用 3,747 个 train 配对。每个 epoch 后会在 dev/test 上对每条音频穷举
101 个候选，并输出：

- `recall_at_1`；
- `recall_at_5`；
- `recall_at_10`；
- 多实体 mention recall；
- 无实体句误报率。

模型以 dev `Recall@5` 选择最佳 checkpoint，而不是使用 test 指标选模型。

训练输出：

```text
../runs/ss_pilot/best.pt
../runs/ss_pilot/last.pt
../runs/ss_pilot/metrics.json
```

## 5. 断点续训

例如从最后一个 checkpoint 继续到 30 epoch：

```bash
uv run python scripts/train_speech_searcher_pilot.py \
  --device cuda \
  --epochs 30 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --num-workers 4 \
  --resume ../runs/ss_pilot/last.pt \
  --output-dir ../runs/ss_pilot
```

## 6. 运行时检查

另开终端观察 GPU：

```bash
watch -n 2 nvidia-smi
```

应看到显存占用和 GPU 利用率。若 GPU 利用率长期很低，可逐步增加 batch size；
若 OOM，则减半。不要修改 train/dev/test 音色划分。

## 7. 需要带回的结果

训练结束后，将以下目录带回本机：

```text
../runs/ss_pilot/
```

至少需要 `best.pt` 和 `metrics.json`。`ss_features` 可以不带回，因为它只是冻结
Whisper 特征缓存，可重新生成。拿到结果后下一步是用 `best.pt` 对 200 条音频生成
真实 SS top-k 候选，再替换 `gl_oracle_seed.jsonl` 中的 oracle candidates。
