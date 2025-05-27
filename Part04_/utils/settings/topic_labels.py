"""
主題標籤設定模組 - 為LDA模型提供各種資料集的自訂主題標籤
"""

# IMDB 電影評論資料集的主題標籤
IMDB_TOPIC_LABELS = {
    # 10個主題
    10: {
        'zh': {
            0: "劇情發展",
            1: "角色塑造",
            2: "演員演技",
            3: "導演手法",
            4: "視覺效果",
            5: "音效配樂",
            6: "情感體驗",
            7: "文化影響",
            8: "批評觀點",
            9: "整體評價"
        },
        'en': {
            0: "Plot Development",
            1: "Character Development",
            2: "Acting Performance",
            3: "Direction",
            4: "Visual Effects",
            5: "Sound & Music",
            6: "Emotional Impact",
            7: "Cultural Impact",
            8: "Critical Analysis",
            9: "Overall Rating"
        }
    }
}

# Amazon 產品評論資料集的主題標籤
AMAZON_TOPIC_LABELS = {
    # 10個主題
    10: {
        'zh': {
            0: "產品品質",
            1: "功能特性",
            2: "使用便利性",
            3: "使用體驗",
            4: "設計外觀",
            5: "耐用程度",
            6: "性價比",
            7: "客戶服務",
            8: "配送物流",
            9: "整體評價"
        },
        'en': {
            0: "Product Quality",
            1: "Features",
            2: "Ease of Use",
            3: "User Experience",
            4: "Design & Aesthetics",
            5: "Durability",
            6: "Value for Money",
            7: "Customer Service",
            8: "Shipping & Logistics",
            9: "Overall Rating"
        }
    }
}

# Yelp 餐廳評論資料集的主題標籤
YELP_TOPIC_LABELS = {
    # 10個主題
    10: {
        'zh': {
            0: "食物品質",
            1: "食物口味",
            2: "菜單選擇",
            3: "服務態度",
            4: "服務速度",
            5: "用餐環境",
            6: "噪音水平",
            7: "位置交通",
            8: "性價比",
            9: "整體評價"
        },
        'en': {
            0: "Food Quality",
            1: "Taste",
            2: "Menu Selection",
            3: "Service Quality",
            4: "Service Speed",
            5: "Ambiance",
            6: "Noise Level",
            7: "Location",
            8: "Value for Money",
            9: "Overall Rating"
        }
    }
}


def get_topic_labels(dataset_name='default', language='zh', num_topics=10):
    """獲取特定資料集的主題標籤
    
    Args:
        dataset_name: 資料集名稱，可以是 'IMDB'、'Amazon'、'Yelp' 或 'default'
        language: 語言，'zh' 為中文，'en' 為英文
        num_topics: 主題數量，支援 5、8、10、15
    
    Returns:
        dict: 主題ID到標籤的映射字典
    """
    # 標準化資料集名稱
    dataset_name = dataset_name.lower()
    
    # 選擇最接近的主題數量
    if num_topics <= 5:
        num_topics = 5
    elif num_topics <= 8:
        num_topics = 8
    elif num_topics <= 10:
        num_topics = 10
    else:
        num_topics = 15
    
    # 根據資料集名稱選擇對應的標籤集
    if 'imdb' in dataset_name or 'movie' in dataset_name:
        labels = IMDB_TOPIC_LABELS.get(num_topics, {}).get(language, {})
    elif 'amazon' in dataset_name or 'product' in dataset_name:
        labels = AMAZON_TOPIC_LABELS.get(num_topics, {}).get(language, {})
    elif 'yelp' in dataset_name or 'restaurant' in dataset_name:
        labels = YELP_TOPIC_LABELS.get(num_topics, {}).get(language, {})
    else:
        labels = DEFAULT_TOPIC_LABELS.get(num_topics, {}).get(language, {})
    
    # 如果找不到合適的標籤，使用默認格式
    if not labels:
        if language == 'zh':
            labels = {i: f"主題 {i+1}" for i in range(num_topics)}
        else:
            labels = {i: f"Topic {i+1}" for i in range(num_topics)}
    
    return labels

def format_topic_labels(topic_dict, add_index=True, separator=': '):
    """格式化主題標籤，增加序號等
    
    Args:
        topic_dict: 主題ID到標籤的映射字典
        add_index: 是否添加序號
        separator: 序號和標籤之間的分隔符
        
    Returns:
        dict: 格式化後的主題ID到標籤的映射字典
    """
    formatted_labels = {}
    
    for idx, label in topic_dict.items():
        if add_index:
            formatted_labels[idx] = f"{idx+1}{separator}{label}"
        else:
            formatted_labels[idx] = label
    
    return formatted_labels

def save_topic_labels_config(config, output_path):
    """保存主題標籤配置到JSON文件
    
    Args:
        config: 主題標籤配置
        output_path: 輸出文件路徑
    """
    import json
    import os
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存配置到JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    return output_path

def convert_topic_key_to_chinese(topic_key, dataset_name='default', num_topics=10):
    """將主題鍵轉換為中文標籤
    
    Args:
        topic_key: 主題鍵，例如 "Topic_3"
        dataset_name: 資料集名稱，用於獲取相應的主題標籤
        num_topics: 主題數量
        
    Returns:
        str: 轉換後的中文主題標籤
    """
    try:
        # 如果已經是中文格式如 "3_主題"，則直接返回
        if not topic_key.startswith("Topic_"):
            return topic_key
        
        # 從主題鍵中提取數字
        parts = topic_key.split("_")
        if len(parts) < 2:
            return topic_key
        
        topic_number = int(parts[1])
        
        # 獲取對應的中文標籤
        labels = get_topic_labels(dataset_name, 'zh', num_topics)
        
        # 查找對應的標籤，索引從0開始，但主題編號從1開始
        topic_label = labels.get(topic_number - 1, f"主題 {topic_number}")
        
        # 返回中文格式 "3_主題標籤"
        return f"{topic_number}_{topic_label}"
    
    except Exception as e:
        import logging
        logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger(__name__)
        logger.error(f"轉換主題鍵時出錯: {str(e)}")
        return topic_key  # 如果出錯，返回原始鍵