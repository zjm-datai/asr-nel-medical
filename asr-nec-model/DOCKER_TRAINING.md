# Docker GPU 训练说明

推荐使用 Docker Compose 以 detached 模式运行 SpeechSearcher 训练。Dev Container
适合交互开发，但训练任务的生命周期可能受到 VS Code 会话影响，因此不作为正式
训练入口。

## 目录结构

服务器上保持以下结构：

```text
workspace/
├── asr-nec-model/       # 本代码仓库
├── data/                # 挂载到容器 /workspace/data
├── weights/base.pt      # 只读挂载到 /workspace/weights
└── runs/                # checkpoint 和日志持久化目录
```

镜像只包含代码与 Python/CUDA 依赖，不包含数据、权重或训练输出。删除容器不会
删除 `data/`、`weights/`和`runs/`。

## 1. 检查服务器

宿主机执行：

```bash
nvidia-smi
docker --version
docker compose version
```

验证 NVIDIA Container Toolkit：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

这条命令必须在容器内显示 GPU。若提示找不到 GPU 或 runtime，应先让管理员安装
NVIDIA Container Toolkit，不能靠修改本项目解决。

## 2. 准备配置

```bash
cd /path/to/workspace/asr-nec-model
cp .env.example .env
mkdir -p ../runs
```

默认使用 GPU 0。如需使用其他卡，修改 `.env`：

```dotenv
CUDA_VISIBLE_DEVICES=1
```

常用显存配置：

| GPU 显存 | FEATURE_BATCH_SIZE | TRAIN_BATCH_SIZE | EVAL_BATCH_SIZE |
| --- | ---: | ---: | ---: |
| 8 GB | 4 | 8–16 | 16–32 |
| 12–16 GB | 8–16 | 16–32 | 32–64 |
| 24 GB 及以上 | 16–32 | 32–64 | 64–128 |

显存不确定时先用默认值；出现 CUDA OOM 后将三个 batch size 减半。

## 3. 构建镜像

```bash
docker compose build ss-train
```

首次构建会下载约数 GB 的 PyTorch CUDA 基础镜像。检查镜像中的 CUDA：

```bash
docker compose run --rm ss-train python -c \
  "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果基础镜像的 CUDA 版本高于宿主机驱动支持范围，在 `.env` 中替换
`PYTORCH_IMAGE`，然后重新构建。例如使用可用的 CUDA 12.4 PyTorch runtime 标签。

## 4. 后台启动

```bash
docker compose up -d ss-train
```

这条命令返回后可以退出 SSH。容器会依次执行：

1. 检查容器内 CUDA；
2. 提取或断点续跑 503 个冻结 Whisper 特征；
3. 训练 SpeechSearcher；
4. 每个 epoch 评估 Recall@1/5/10；
5. 持久化最佳和最后 checkpoint。

## 5. 监控

```bash
docker compose ps -a
docker compose logs -f --tail=100 ss-train
watch -n 2 nvidia-smi
```

宿主机日志同时写入：

```text
../runs/ss_pilot/container.log
```

训练结果：

```text
../runs/ss_pilot/best.pt
../runs/ss_pilot/last.pt
../runs/ss_pilot/metrics.json
```

容器状态为 `Exited (0)` 表示训练正常结束，不表示失败。查看最终日志确认
`SpeechSearcher container job completed`。

## 6. 停止与续训

优雅停止：

```bash
docker compose stop -t 60 ss-train
```

脚本每个 epoch 保存 `last.pt`。重新启动时 `AUTO_RESUME=1` 会自动加载它：

```bash
docker compose up -d ss-train
```

注意：若在一个 epoch 中途停止，只能从上一个完整 epoch 恢复。

要将总 epoch 从20提高到30，修改 `.env`：

```dotenv
EPOCHS=30
AUTO_RESUME=1
```

然后启动同一个 Compose 服务。

## 7. 开始全新实验

不要在旧 checkpoint 上设置 `AUTO_RESUME=0`后直接覆盖同一个输出目录。建议先把
旧结果移动到新的实验目录，或修改 Compose/启动脚本中的 `RUN_DIR`。当前先导实验
只维护一个 `runs/ss_pilot`目录。

## 8. 常用故障检查

容器看不到 GPU：

```bash
docker info | grep -i runtime
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

查看退出码：

```bash
docker inspect ss-train --format '{{.State.ExitCode}} {{.State.Error}}'
```

进入容器检查挂载：

```bash
docker compose run --rm --entrypoint bash ss-train
ls -lh ../weights/base.pt
ls -lh ../data/speech_searcher/audio_pilot/pilot_manifest.jsonl
```

重新构建代码镜像不会删除宿主机 checkpoint：

```bash
docker compose build --no-cache ss-train
docker compose up -d ss-train
```
