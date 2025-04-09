import torch
import platform
import sys
import subprocess
import os
# 嘗試啟用 CUDA 向下相容模式
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

def run_command(cmd):
    """執行系統命令並返回輸出"""
    try:
        result = subprocess.run(cmd, shell=True, check=False, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               text=True)
        return result.stdout + result.stderr
    except Exception as e:
        return f"執行命令出錯: {str(e)}"

def detailed_cuda_check():
    """詳細的CUDA檢查"""
    print("="*40)
    print("CUDA 環境診斷報告")
    print("="*40)
    
    # 操作系統資訊
    print(f"操作系統: {platform.system()} {platform.version()}")
    print(f"Python版本: {sys.version}")
    
    # PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"PyTorch CUDA版本: {torch.version.cuda}")
    
    # CUDA可用性
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA設備數量: {torch.cuda.device_count()}")
        print(f"當前CUDA設備: {torch.cuda.current_device()}")
        print(f"CUDA設備名稱: {torch.cuda.get_device_name(0)}")
        try:
            print(f"CUDA能力: {torch.cuda.get_device_capability(0)}")
        except:
            print("無法獲取CUDA能力")
    
    # 檢查NVIDIA驅動程序
    print("\n-- NVIDIA-SMI 輸出 --")
    print(run_command("nvidia-smi"))
    
    # 嘗試進行簡單的CUDA操作
    print("\n-- CUDA運算測試 --")
    if torch.cuda.is_available():
        try:
            # 建立一個小的CUDA張量
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            y = torch.tensor([4.0, 5.0, 6.0], device='cuda')
            z = x + y
            print(f"CUDA張量測試 - 成功: {z.cpu().numpy()}")
            
            # 嘗試運行一個小的神經網絡
            model = torch.nn.Linear(3, 1).cuda()
            input_data = torch.randn(1, 3).cuda()
            output = model(input_data)
            print(f"神經網絡測試 - 成功: {output.shape}")
            
            # 嘗試生成特定的CUDA核心
            a = torch.randn(100, 100, device='cuda')
            b = torch.randn(100, 100, device='cuda')
            c = torch.matmul(a, b)
            print(f"矩陣乘法測試 - 成功: {c.shape}")
            
        except Exception as e:
            print(f"CUDA測試失敗: {str(e)}")
    else:
        print("CUDA不可用，跳過測試")
    
    # 環境變數
    print("\n-- CUDA相關環境變數 --")
    for var in ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES']:
        print(f"{var}: {os.environ.get(var, '未設置')}")
    
    print("="*40)
    print("診斷完成")
    print("="*40)

def basic_cpu_test():
    """在CPU上進行基本測試以確認PyTorch功能正常"""
    print("\n-- CPU運算測試 --")
    try:
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        z = x + y
        print(f"CPU張量測試 - 成功: {z.numpy()}")
        return True
    except Exception as e:
        print(f"CPU測試失敗: {str(e)}")
        return False
def check_cuda_versions():
    """檢查系統上的 CUDA 版本"""
    print("\n-- 系統 CUDA 版本檢查 --")
    
    # 檢查 nvcc 版本
    nvcc_output = run_command("nvcc --version")
    print("NVCC 版本信息:")
    print(nvcc_output)
    
    # 檢查 CUDA 安裝目錄
    cuda_dirs = []
    base_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"
    if os.path.exists(base_path):
        for dir_name in os.listdir(base_path):
            if dir_name.startswith('v'):
                cuda_dirs.append(dir_name)
    
    print(f"發現的 CUDA 安裝目錄: {cuda_dirs}")
    
    # 檢查 PATH 環境變數中的 CUDA 路徑
    path_var = os.environ.get('PATH', '')
    cuda_paths = [p for p in path_var.split(os.pathsep) if 'CUDA' in p]
    print("PATH 環境變數中的 CUDA 路徑:")
    for p in cuda_paths:
        print(f"  - {p}")

if __name__ == "__main__":
    check_cuda_versions()
    detailed_cuda_check()
    basic_cpu_test()