from langchain.memory import ConversationBufferWindowMemory
from datetime import datetime, timedelta
from adaptive_rag.utils.state import AdaptiveRagState
# 유저별 메모리와 마지막 활동 시각 저장소
memory_store: dict[str, ConversationBufferWindowMemory] = {}
last_activity: dict[str, datetime] = {}

# 세션 유지 시간 
SESSION_TIMEOUT = timedelta(minutes=5)

# 유지할 최근 메시지 수
WINDOW_SIZE = 3

def get_user_memory(user_id: str) -> ConversationBufferWindowMemory:
    now = datetime.now()

    # 초기화 조건: 세션이 없거나 5분 이상 경과 시
    if (
        user_id not in memory_store or
        user_id not in last_activity or
        now - last_activity[user_id] > SESSION_TIMEOUT
    ):
        memory_store[user_id] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=WINDOW_SIZE  # 최근 k개 메시지만 유지
        )

    # 활동 시간 갱신
    last_activity[user_id] = now

    return memory_store[user_id]