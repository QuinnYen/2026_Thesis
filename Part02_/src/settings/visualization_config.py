# visualization_config.py
# 設定 matplotlib 支援中文顯示
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging

def configure_chinese_fonts():
    """配置 matplotlib 支援中文顯示"""
    try:
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 
                                          'STHeiti', 'HanaMinA', 'WenQuanYi Micro Hei', 'PingFang TC', 
                                          'Hiragino Sans GB', 'Heiti TC']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 測試中文顯示
        test_font = plt.rcParams['font.sans-serif'][0]
        return f"中文字體配置成功，使用字體: {test_font}"
    except Exception as e:
        logging.error(f"配置中文字體時出錯: {str(e)}")
        return f"中文字體配置失敗: {str(e)}"

# 在模組載入時自動配置
config_result = configure_chinese_fonts()
logging.info(config_result)