# 情感分析主顋標籤
TOPIC_LABELS = {
    # IMDB電影評論的主題標籤
    "imdb": {
        0: "Viewing Experience",       # 觀影體驗
        1: "Heroic Elements",          # 英雄故事元素 
        2: "Female Roles & Comedy",    # 女性角色與喜劇
        3: "Overall Evaluation",       # 整體評價
        4: "Production Elements",      # 製作元素
        5: "Audience Value",           # 觀眾價值感受
        6: "Actors & Fan Response",    # 演員與粉絲反應
        7: "Character & Setting",      # 角色與環境
        8: "Stars & Storytelling",     # 明星與故事
        9: "Personal Preferences"      # 個人喜好
    },
    
    # Amazon產品評論的主題標籤 (預留)
    "amazon": {
        # 預留給未來 Amazon 資料分析後填入
        # 例如：0: "Product Quality", 1: "Price Value" 等
    },
    
    # Yelp餐廳評論的主題標籤 (預留)
    "yelp": {
        # 預留給未來 Yelp 資料分析後填入
        # 例如：0: "Food Quality", 1: "Service Experience" 等
    }
}