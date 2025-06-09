import streamlit as st
import requests
import uuid
import os

# 이 파일(app.py) 위치 기준
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 페이지 설정
st.set_page_config(
    page_title="마이폴리오 챗봇",
    page_icon=os.path.join(BASE_DIR, "asset", "mypolio.png"),
    layout="centered",
)

# CSS 로드
def load_css(file_name: str):
    css_path = os.path.join(BASE_DIR, file_name)
    if not os.path.isfile(css_path):
        raise FileNotFoundError(f"CSS 파일을 찾을 수 없습니다: {css_path}")
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --- Logo and Title ---
# Centered logo and initial message
try:
    # 간단하고 확실한 중앙 정렬 방법
    st.markdown("""
    <div style='
        text-align: center;
        width: 100%;
        margin: 2rem 0;
    '>
    """, unsafe_allow_html=True)
    
    # 로고를 중앙에 배치
    logo_path = os.path.join(BASE_DIR, "asset", "mypolio.png")
    col1, col2, col3 = st.columns([1.7, 1, 1.5])
    with col2:
        st.image(logo_path, width=100)
    
    # 제목들도 중앙 정렬로 배치
    st.markdown("""
        <h1 style='color: #FFFFFF; font-size: 24px; font-weight: bold; margin: 1rem 0 0.5rem 0; text-align: center;'>안녕하세요!</h1>
        <h2 style='color: #FFFFFF; font-size: 18px; margin: 0 0 2rem 0; text-align: center;'>마이폴리오 AI 챗봇입니다</h2>
    </div>
    """, unsafe_allow_html=True)
    
except FileNotFoundError:
    st.warning("로고 이미지를 \'frontend/asset/mypolio.png\'에서 찾을 수 없습니다.")

# Remove the default Streamlit title as we are using markdown for title
# st.title("Bearable Chatbot") 


# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "stage" not in st.session_state:
    st.session_state.stage = "initial"  # Stages: initial, main_selected, sub_selected, chatting
if "category" not in st.session_state:
    st.session_state.category = None

# --- Button Interaction Logic ---

categories =  ["운영문의", "과목문의", "입시문의", "도서문의", "서비스 문의"]


def handle_category_selection(category_name):
    st.session_state.category = category_name
    # 서비스 문의인 경우 외부 링크 안내 후 초기 상태로 복귀
    if category_name == "서비스 문의":
        st.session_state.messages.append({"role": "user", "content": category_name})
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"'{category_name}' 관련 문의는 [링크](https://myfolio.im/contents/collegeInfo)하단의 채널톡을 이용해주세요."
        })
        # 새 탭으로 열기 스크립트
        js_open = "<script>window.open('https://myfolio.im/contents/collegeInfo','_blank');</script>"
        st.markdown(js_open, unsafe_allow_html=True)
        st.session_state.stage = "initial"
    else:
        st.session_state.stage = "chatting"
        st.session_state.messages.append({"role": "user", "content": category_name})
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"'{category_name}'에 대해 궁금한 점을 입력해주세요."
        })
    st.rerun()

# --- Message Display and Interaction Logic ---

# Display initial greeting (now part of logo/title section)
if st.session_state.stage == "initial":
    if not st.session_state.messages:
        # The initial "안녕하세요! 마이폴리오 AI 챗봇입니다" is now above the buttons.
        # The first assistant message in chat will be "무엇을 도와드릴까요?"
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "무엇을 도와드릴까요?"
        })

    # Display chat messages (which will include the "무엇을 도와드릴까요?")
    for message in st.session_state.messages:
        avatar_img = logo_path if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar_img):
            st.markdown(message["content"])
    
    # Main category buttons
    cols = st.columns(len(categories))
    for i, category in enumerate(categories):
        if cols[i].button(category, key=f"{category}", use_container_width=True):
            handle_category_selection(category)
    st.stop()

# 채팅 화면: 메시지 표시 + 입력
if st.session_state.stage == "chatting":
    for msg in st.session_state.messages:
        avatar = os.path.join(BASE_DIR, "asset", "mypolio.png") if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("documents"):
                with st.expander("참고 문서 보기", expanded=False):
                    for doc in msg["documents"]:
                        source = doc.get("metadata", {}).get("source", "N/A")
                        content = doc.get("page_content", "")
                        st.markdown(f"**출처:** {source}")
                        st.caption(content)
                        st.divider()


    # 사용자 입력
    prompt = st.chat_input("마이폴리오 AI에게 무엇이든 물어보세요!", disabled=False)
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            payload = {
                "question": prompt,
                "user_id": st.session_state.user_id,
                "category": st.session_state.category
            }
            res = requests.post("http://localhost:8000/chat/", json=payload)
            res.raise_for_status()
            data = res.json()
            answer = data.get("answer", "죄송합니다. 응답을 받지 못했습니다.")
            docs = data.get("documents", [])
            assistant_msg = {"role": "assistant", "content": answer}
            if docs:
                assistant_msg["documents"] = docs
            st.session_state.messages.append(assistant_msg)

        except requests.exceptions.RequestException as e:
            st.error(f"백엔드 연결 오류: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"오류: 백엔드 연결에 실패했습니다. {e}"})
        except Exception as e:
            st.error(f"예상치 못한 오류 발생: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"오류: 예상치 못한 오류가 발생했습니다. {e}"})
    
        st.rerun()
