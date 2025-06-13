# BERT情感分析系統 Docker部署指南

## 🎯 概述

本文檔說明如何在Docker環境中部署BERT情感分析系統，支援GPU加速和CPU運行模式。

## 📋 系統需求

### GPU版本
- Docker >= 20.10
- docker-compose >= 1.28
- NVIDIA Docker Runtime (nvidia-docker2)
- CUDA 11.8+
- GPU記憶體 >= 4GB

### CPU版本
- Docker >= 20.10
- docker-compose >= 1.28
- 記憶體 >= 8GB

## 🚀 快速開始

### 1. 安裝Docker和NVIDIA Runtime（GPU版本）

```bash
# Ubuntu/Debian
# 安裝Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安裝NVIDIA Docker Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. 部署系統

```bash
# 下載專案
git clone <your-repo-url>
cd 2026_Thesis

# 執行部署腳本
chmod +x deploy.sh
./deploy.sh
```

### 3. 使用系統

```bash
# 查看容器狀態
docker-compose ps

# 進入容器（GPU版本）
docker-compose exec bert-analysis-gpu bash

# 進入容器（CPU版本）
docker-compose exec bert-analysis-cpu bash

# 運行分析
python Part05_Main.py --classify data.csv
```

## 📊 服務管理

### 啟動服務

```bash
# GPU版本
docker-compose up -d bert-analysis-gpu

# CPU版本
docker-compose up -d bert-analysis-cpu

# 開發環境
docker-compose up -d bert-dev
```

### 停止服務

```bash
docker-compose down
```

### 查看日誌

```bash
docker-compose logs -f bert-analysis-gpu
```

### 重建映像

```bash
docker-compose build --no-cache
```

## 📁 數據卷配置

系統會掛載以下目錄：

- `./data` → `/app/data`：輸入數據
- `./output` → `/app/output`：分析結果
- `./models` → `/app/models`：模型文件
- `./logs` → `/app/logs`：日誌文件

## 🎮 GPU使用說明

### 檢查GPU狀態

```bash
# 在容器內
nvidia-smi

# 檢查PyTorch GPU支援
python -c "import torch; print(f'GPU可用: {torch.cuda.is_available()}')"
```

### GPU記憶體管理

```bash
# 設置GPU記憶體限制
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 🔧 常見問題解決

### 1. GPU不可用

```bash
# 檢查NVIDIA Driver
nvidia-smi

# 檢查Docker GPU支援
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### 2. 記憶體不足

```bash
# 清理Docker緩存
docker system prune -a

# 增加swap空間
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 3. 網路問題

```bash
# 使用國內鏡像
export DOCKER_BUILDKIT=1
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ .
```

## 📈 性能優化

### GPU優化

```bash
# 設置環境變數
export CUDA_VISIBLE_DEVICES=0
export TORCH_BACKENDS_CUDNN_BENCHMARK=true
export PYTHONPATH=/app:$PYTHONPATH
```

### 內存優化

```bash
# 限制記憶體使用
docker-compose run --memory="8g" bert-analysis-gpu
```

## 🔍 監控和診斷

### 資源監控

```bash
# 實時監控
docker stats

# GPU監控
watch -n 1 nvidia-smi
```

### 診斷命令

```bash
# 檢查容器健康狀態
docker-compose exec bert-analysis-gpu python -c "
import torch, transformers, sklearn, pandas
print('✅ 所有依賴正常')
print(f'✅ GPU可用: {torch.cuda.is_available()}')
"
```

## 📞 技術支援

如有問題，請查看：

1. 容器日誌：`docker-compose logs -f`
2. 系統狀態：`docker-compose ps`
3. 資源使用：`docker stats`
4. GPU狀態：`nvidia-smi` 