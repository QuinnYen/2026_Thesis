# deploy.ps1
# 檢查 Docker 是否安裝
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker 未安裝，請先安裝 Docker Desktop"
    exit 1
}

# 檢查 docker-compose 是否安裝
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Error "docker-compose 未安裝，請先安裝 Docker Desktop"
    exit 1
}

# 檢查 GPU 支持
$hasGpu = $false
try {
    $nvidiaSmi = docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
    if ($nvidiaSmi) {
        $hasGpu = $true
        Write-Host "✅ 檢測到 GPU 支持"
    }
} catch {
    Write-Host "ℹ️ 未檢測到 GPU 支持，將使用 CPU 版本"
}

# 創建必要的目錄
$directories = @("data", "output", "models", "logs")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir
        Write-Host "✅ 創建目錄: $dir"
    }
}

# 構建和運行容器
if ($hasGpu) {
    Write-Host "🚀 正在部署 GPU 版本..."
    docker-compose -f docker/docker-compose.yml up -d bert-analysis-gpu
} else {
    Write-Host "🚀 正在部署 CPU 版本..."
    docker-compose -f docker/docker-compose.yml up -d bert-analysis-cpu
}

# 顯示容器狀態
Write-Host "`n📊 容器狀態："
docker-compose -f docker/docker-compose.yml ps

# 顯示使用說明
Write-Host "`n📝 使用說明："
if ($hasGpu) {
    Write-Host "1. 將數據文件放入 data/ 目錄"
    Write-Host "2. 執行分析：docker-compose -f docker/docker-compose.yml exec bert-analysis-gpu python Part05_Main.py"
    Write-Host "3. 查看結果：output/ 目錄"
} else {
    Write-Host "1. 將數據文件放入 data/ 目錄"
    Write-Host "2. 執行分析：docker-compose -f docker/docker-compose.yml exec bert-analysis-cpu python Part05_Main.py"
    Write-Host "3. 查看結果：output/ 目錄"
}