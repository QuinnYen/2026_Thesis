"""
可視化配置中文模組
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import platform
import os
import sys
from matplotlib.font_manager import FontProperties

# 使用非互動模式，避免與Tkinter衝突
plt.ioff()

logger = logging.getLogger(__name__)

# 根據不同操作系統設定較優先的中文字體
def get_system_fonts():
    """根據作業系統取得適合的中文字體列表"""
    system = platform.system()
    
    if system == 'Windows':
        return ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
    elif system == 'Darwin':  # macOS
        return ['PingFang TC', 'STHeiti', 'Heiti TC', 'Hiragino Sans GB', 'Apple LiGothic', 'Apple LiSung']
    else:  # Linux 和其他系統
        return ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK TC', 'Noto Sans CJK SC', 'HanaMinA', 'AR PL UMing CN']

def configure_chinese_fonts():
    """配置 matplotlib 支援中文顯示"""
    try:
        # 獲取系統特定的字體
        system_fonts = get_system_fonts()
        
        # 一般使用的字體優先順序
        general_fonts = system_fonts + [
            'Arial Unicode MS', 'DejaVu Sans', 'FreeSans', 'Droid Sans Fallback',
            'Source Han Sans TC', 'Source Han Sans SC', 'Noto Sans CJK JP'
        ]
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = general_fonts
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 檢查是否有可用的中文字體
        found_font = False
        for font in general_fonts:
            try:
                fp = FontProperties(family=font)
                if fp.get_name() != 'DejaVu Sans':  # 如果沒有找到font，matplotlib會回傳默認字體
                    found_font = True
                    logger.info(f"找到可用的中文字體: {font}")
                    break
            except:
                continue
        
        if not found_font:
            logger.warning("沒有找到系統支援的中文字體，可能會導致中文顯示不正確")
        
        # 返回配置結果
        return f"中文字體配置完成，嘗試使用字體優先順序: {', '.join(general_fonts[:3])}"
    except Exception as e:
        logger.error(f"配置中文字體時出錯: {str(e)}")
        return f"中文字體配置失敗: {str(e)}"

def apply_chinese_to_plot(ax, title=None, xlabel=None, ylabel=None, fontsize_title=14, fontsize_label=12):
    """
    將中文標題和標籤應用到圖表
    
    Args:
        ax: matplotlib軸對象
        title: 圖表標題
        xlabel: x軸標籤
        ylabel: y軸標籤
        fontsize_title: 標題字體大小
        fontsize_label: 標籤字體大小
    """
    try:
        if title:
            ax.set_title(title, fontsize=fontsize_title)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize_label)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize_label)
        
        # 確保刻度標籤也能正確顯示中文
        plt.tick_params(labelsize=fontsize_label-2)
    except Exception as e:
        logger.error(f"應用中文設置失敗: {str(e)}")
        plt.close('all')  # 確保錯誤時清理資源
        raise

def format_topic_labels(topic_dict, add_index=True):
    """
    格式化主題標籤用於可視化
    
    Args:
        topic_dict: 主題標籤字典，如 TOPIC_LABELS["imdb"]
        add_index: 是否在標籤前添加索引編號
        
    Returns:
        dict: 格式化後的主題標籤字典
    """
    formatted_labels = {}
    
    for idx, label in topic_dict.items():
        # 檢查標籤是否有中文註解（以#開頭的註解）
        if '#' in label:
            label_parts = label.split('#', 1)
            english_label = label_parts[0].strip()
            chinese_note = label_parts[1].strip()
            
            if add_index:
                formatted_labels[idx] = f"{idx+1}. {english_label}\n({chinese_note})"
            else:
                formatted_labels[idx] = f"{english_label}\n({chinese_note})"
        else:
            # 如果沒有註解，就直接使用標籤
            if add_index:
                formatted_labels[idx] = f"{idx+1}. {label}"
            else:
                formatted_labels[idx] = label
    
    return formatted_labels

def check_chinese_display():
    """
    檢查中文顯示是否正常，並生成測試圖
    
    Returns:
        bool: 中文顯示是否正常
    """
    try:
        # 確保使用非互動模式
        plt.ioff()
        
        # 創建一個簡單的測試圖
        plt.figure(figsize=(10, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
        plt.title('中文標題測試')
        plt.xlabel('橫軸標籤')
        plt.ylabel('縱軸標籤')
        plt.annotate('註釋文字', xy=(2, 4))
        plt.text(3, 10, '文字測試')
        
        # 保存測試圖
        test_dir = './Part02_/logs'
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            
        # 保存並立即關閉圖表
        test_path = os.path.join(test_dir, 'chinese_font_test.png')
        plt.savefig(test_path)
        plt.close('all')  # 確保關閉所有圖表
        
        logger.info(f"中文顯示測試圖已保存至: {test_path}")
        return True
    except Exception as e:
        plt.close('all')  # 錯誤時也要清理
        logger.error(f"中文顯示測試失敗: {str(e)}")
        return False

# 在模組載入時自動配置，但避免在導入時進行驗證測試
config_result = configure_chinese_fonts()
logger.info(config_result)