from pymongo import MongoClient
from datetime import datetime
import json
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os
from adaptive_rag.utils.state import AdaptiveRagState

# API 키 정보 로드
load_dotenv()

# API 키 읽어오기
MONGODB_URI = os.environ.get("MONGODB_URI")

# MongoDB Atlas 연결
client = MongoClient(MONGODB_URI)
db = client["chatbot_db"]
collection = db["chat_logs"]

def save_chat_log(user_input, bot_response, category="미지정", user_id = "anonymous"):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user_input,
        "bot": bot_response,
        "category": category,
        "user_id": user_id
    }
    collection.insert_one(log_entry)  # <- 이게 저장하는 코드