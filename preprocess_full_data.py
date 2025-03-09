
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import datetime

print("Starting full data preprocessing...")

# 確保必要的NLTK資源已下載
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# 預處理選項
CLEAN_HTML = True
REMOVE_PUNCTUATION = True
LOWERCASE = True
REMOVE_STOPWORDS = True
STEMMING = False
LEMMATIZATION = True

# 準備預處理工具
stop_words = set(stopwords.words('english')) if REMOVE_STOPWORDS else set()
stemmer = PorterStemmer() if STEMMING else None
lemmatizer = WordNetLemmatizer() if LEMMATIZATION else None

def preprocess_text(text):
    processed_text = text

    # 執行各預處理步驟
    if CLEAN_HTML:
        processed_text = re.sub(r'<.*?>', '', processed_text)

    if LOWERCASE:
        processed_text = processed_text.lower()

    if REMOVE_PUNCTUATION:
        processed_text = re.sub(r'[^\w\s]', '', processed_text)

    # 分詞
    tokens = processed_text.split()

    if REMOVE_STOPWORDS:
        tokens = [token for token in tokens if token not in stop_words]

    if STEMMING:
        tokens = [stemmer.stem(token) for token in tokens]

    if LEMMATIZATION:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # 重新組合文本
    return ' '.join(tokens)

# 處理IMDB資料
try:
    imdb_path = r"D:/Project/2026_Thesis/ReviewsDataBase/IMDB Dataset.csv"
    if os.path.exists(imdb_path):
        print(f"Processing IMDB data: {imdb_path}")
        imdb_data = pd.read_csv(imdb_path)

        review_col = 'review' if 'review' in imdb_data.columns else 'text'
        if review_col in imdb_data.columns:
            print(f"Found review column: {review_col}, total {len(imdb_data)} records")

            # 處理評論文本
            imdb_data['preprocessed_review'] = imdb_data[review_col].astype(str).apply(preprocess_text)

            # 儲存處理結果
            output_file = f"imdb_preprocessed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            imdb_data.to_csv(output_file, index=False)
            print(f"IMDB data processing completed, saved to: {output_file}")
    else:
        print(f"IMDB data file not found: {imdb_path}")
except Exception as e:
    print(f"Error processing IMDB data: {str(e)}")

print("Full data preprocessing completed")
input("Press any key to exit...")
