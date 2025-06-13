#!/bin/bash
# BERT情感分析系統 Docker部署腳本

set -e

echo "🚀 BERT情感分析系統 Docker部署腳本"
echo "=================================="

# 檢查Docker和docker-compose
check_dependencies() {
    echo "🔍 檢查系統依賴..."
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker未安裝，請先安裝Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ docker-compose未安裝，請先安裝docker-compose"
        exit 1
    fi
    
    echo "✅ Docker和docker-compose已安裝"
}

# 檢查GPU支援
check_gpu() {
    echo "🎮 檢查GPU支援..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ 檢測到NVIDIA GPU"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        GPU_AVAILABLE=true
    else
        echo "⚠️  未檢測到NVIDIA GPU，將使用CPU版本"
        GPU_AVAILABLE=false
    fi
}

# 創建必要目錄
create_directories() {
    echo "📁 創建必要目錄..."
    mkdir -p data output models logs
    echo "✅ 目錄創建完成"
}

# 構建和運行
deploy() {
    echo "🔨 開始構建Docker映像..."
    
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "🎮 構建GPU版本..."
        docker-compose build bert-analysis-gpu
        echo "🚀 啟動GPU版本..."
        docker-compose up -d bert-analysis-gpu
    else
        echo "💻 構建CPU版本..."
        docker-compose build bert-analysis-cpu
        echo "🚀 啟動CPU版本..."
        docker-compose up -d bert-analysis-cpu
    fi
    
    echo "✅ 部署完成！"
}

# 顯示狀態
show_status() {
    echo "📊 容器狀態:"
    docker-compose ps
    
    echo ""
    echo "📋 使用說明:"
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "   • GPU版本: http://localhost:8080"
        echo "   • 進入容器: docker-compose exec bert-analysis-gpu bash"
    else
        echo "   • CPU版本: http://localhost:8081"
        echo "   • 進入容器: docker-compose exec bert-analysis-cpu bash"
    fi
    echo "   • 查看日誌: docker-compose logs -f"
    echo "   • 停止服務: docker-compose down"
}

# 主程式
main() {
    check_dependencies
    check_gpu
    create_directories
    deploy
    show_status
}

# 執行主程式
main "$@" 