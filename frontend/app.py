import streamlit as st
import uuid
import os

# RAG pipeline import
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from adaptive_rag.utils.pipeline import initialize_graph_for_api, get_chatbot_response

# --- 초기화: RAG 그래프를 메모리에 올려둡니다 ---
initialize_graph_for_api()

# 앱 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(
    page_title="마이폴리오 챗봇",
    page_icon=os.path.join(BASE_DIR, "asset", "mypolio.png"),
    layout="centered",
)

# CSS
def load_css(file_name: str):
    css_path = os.path.join(BASE_DIR, file_name)
    if os.path.isfile(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS 파일을 찾을 수 없습니다: {css_path}")

load_css("style.css")

# 로고+타이틀
logo_path = os.path.join(BASE_DIR, "asset", "mypolio.png")
try:
    st.markdown(f"""
    <div style="text-align:center; margin:2rem 0;">
      <img src="{logo_path}" width="100"/>
      <h1 style="color:#FFF; font-size:24px; font-weight:bold;">안녕하세요!</h1>
      <h2 style="color:#FFF; font-size:18px;">마이폴리오 AI 챗봇입니다</h2>
    </div>
    """, unsafe_allow_html=True)
except:
    st.warning("로고 이미지를 불러올 수 없습니다.")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "stage" not in st.session_state:
    st.session_state.stage = "initial"  # initial, chatting
if "category" not in st.session_state:
    st.session_state.category = None

# 카테고리 목록
categories = ["운영문의", "과목문의", "입시문의", "도서문의", "서비스 문의"]

# 카테고리 선택 핸들러
def handle_category(cat: str):
    st.session_state.category = cat
    st.session_state.messages.append({"role":"user", "content":cat})
    if cat == "서비스 문의":
        st.session_state.messages.append({
            "role":"assistant",
            "content":"‘서비스 문의’ 관련은 아래 링크 하단 채널톡을 이용해주세요:\nhttps://myfolio.im/contents/collegeInfo"
        })
        st.session_state.stage = "initial"
    else:
        st.session_state.messages.append({
            "role":"assistant",
            "content":f"‘{cat}’에 대해 궁금하신 점을 입력해주세요."
        })
        st.session_state.stage = "chatting"
    st.experimental_rerun()

# 초기 화면: 인사 + 카테고리 버튼
if st.session_state.stage == "initial":
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role":"assistant", "content":"무엇을 도와드릴까요?"
        })
    for msg in st.session_state.messages:
        avatar = logo_path if msg["role"]=="assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
    cols = st.columns(len(categories))
    for i, cat in enumerate(categories):
        if cols[i].button(cat, use_container_width=True):
            handle_category(cat)
    st.stop()

# 채팅 화면: 기존 메시지 + 입력창
for msg in st.session_state.messages:
    avatar = logo_path if msg["role"]=="assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
prompt = st.chat_input("마이폴리오 AI에게 무엇이든 물어보세요!", disabled=False)
if prompt:
    # 유저 메시지 기록
    st.session_state.messages.append({"role":"user","content":prompt})

    # RAG 파이프라인 호출
    result = get_chatbot_response(prompt,
                                  st.session_state.user_id,
                                  st.session_state.category)
    if "error" in result:
        reply = f"오류 발생: {result['error']}"
    else:
        reply = result.get("generation", "죄송해요, 답변을 못 찾았어요.")
        # 문서 프리뷰가 있으면 나중에 expander로 추가할 수 있습니다

    # 어시스턴트 응답 기록
    st.session_state.messages.append({"role":"assistant","content":reply})
    st.experimental_rerun()
