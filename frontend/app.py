import streamlit as st
import requests
import uuid

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="마이폴리오 챗봇",
    page_icon="asset/mypolio.png",  # Path relative to the app.py file
    layout="centered"
)

# --- Custom CSS Injection from External File ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css") # Assuming style.css is in the same directory as app.py

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
    col1, col2, col3 = st.columns([1.7, 1, 1.5])
    with col2:
        st.image("asset/mypolio.png", width=100)
    
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
if "main_category" not in st.session_state:
    st.session_state.main_category = None
if "sub_category" not in st.session_state:
    st.session_state.sub_category = None

# --- Button Interaction Logic ---

main_categories = {
    "고교학점제": ["과목문의", "운영방침", "기타"],
    "진학상담": ["대학선택", "학과선택", "기타"],
    "서비스 문의": ["이용방법", "오류신고", "기타"]
}

def handle_main_category_selection(category_name):
    st.session_state.main_category = category_name
    st.session_state.stage = "main_selected"
    # Add user's choice as a message
    st.session_state.messages.append({"role": "user", "content": category_name})
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"'{st.session_state.main_category}' 문의 내용을 아래에서 선택해주세요."
    })
    st.rerun()

def handle_sub_category_selection(sub_category_name):
    st.session_state.sub_category = sub_category_name
    st.session_state.stage = "sub_selected"
    # Add user's choice as a message
    st.session_state.messages.append({"role": "user", "content": sub_category_name})
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"'{st.session_state.main_category} > {st.session_state.sub_category}'에 대해 궁금한 점을 질문해주세요."
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
        avatar_img = "asset/mypolio.png" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar_img):
            st.markdown(message["content"])
    
    # Main category buttons
    cols = st.columns(len(main_categories))
    for i, (category, _) in enumerate(main_categories.items()):
        if cols[i].button(category, key=f"main_{category}", use_container_width=True):
            handle_main_category_selection(category)
    st.stop()

elif st.session_state.stage == "main_selected":
    # Display chat messages
    for message in st.session_state.messages:
        avatar_img = "asset/mypolio.png" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar_img):
            st.markdown(message["content"])

    sub_categories = main_categories[st.session_state.main_category]
    
    # Sub-category buttons - wrapped in a div for specific styling
    st.markdown('<div class="sub-category-buttons">', unsafe_allow_html=True)
    cols = st.columns(len(sub_categories))
    for i, sub_cat in enumerate(sub_categories):
        if st.session_state.main_category == "서비스 문의":
            if cols[i].button(sub_cat, key=f"service_sub_{st.session_state.main_category}_{sub_cat}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": sub_cat}) # Add user choice
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"'{st.session_state.main_category} > {sub_cat}' 관련 문의는 [링크](https://myfolio.im/seteuk)를 통해 확인해주세요. 해당 링크가 새 탭으로 열립니다."
                })
                js_open_new_tab = "<script>window.open('https://myfolio.im/seteuk', '_blank');</script>"
                st.markdown(js_open_new_tab, unsafe_allow_html=True)
                # No st.rerun() here, let the message display and then user can ask something else or select again.
                # Or, if we want to go back to initial after this, we can set stage and rerun.
                # For now, let it stay to show the link message.
                st.session_state.stage = "chatting" # Allow further interaction or new selections
                st.rerun() # Rerun to show the new messages and update state
        else:
            if cols[i].button(sub_cat, key=f"sub_{st.session_state.main_category}_{sub_cat}", use_container_width=True):
                handle_sub_category_selection(sub_cat)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Consolidating the message display loop for stages where chat happens
if st.session_state.stage in ["sub_selected", "chatting"]:
    for message in st.session_state.messages:
        avatar_img = "asset/mypolio.png" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar_img):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "documents" in message and message["documents"]:
                with st.expander("참고 문서 보기", expanded=False):
                    for doc_item in message["documents"]:
                        source = "N/A"
                        page_content = ""
                        if isinstance(doc_item, dict): 
                            source = doc_item.get("metadata", {}).get("source", "N/A")
                            page_content = doc_item.get("page_content", "")
                        elif hasattr(doc_item, 'metadata') and hasattr(doc_item, 'page_content'):
                            source = doc_item.metadata.get('source', 'N/A') if isinstance(doc_item.metadata, dict) else "N/A"
                            page_content = doc_item.page_content
                        
                        st.markdown(f"**출처:** {source}")
                        st.caption(page_content)
                        st.divider()

# React to user input via st.chat_input
chat_input_disabled = st.session_state.stage not in ["sub_selected", "chatting"]
# Placeholder text to match the image
chat_input_placeholder = "마이폴리오 AI에게 무엇이든 물어보세요!" if not chat_input_disabled else "먼저 상단에서 카테고리를 선택해주세요."

if prompt := st.chat_input(chat_input_placeholder, disabled=chat_input_disabled, key="chat_input_main"):
    st.session_state.stage = "chatting"
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message immediately
    # with st.chat_message("user"):
    #     st.markdown(prompt)

    try:
        # Backend API call
        response = requests.post(
            "http://localhost:8000/chat/",
            json={"question": prompt, "user_id": st.session_state.user_id, 
                    "main_category": st.session_state.main_category, 
                    "sub_category": st.session_state.sub_category}
        )
        response.raise_for_status()
        response_data = response.json()
        answer = response_data.get("answer", "죄송합니다, 답변을 받아오지 못했습니다.")
        documents = response_data.get("documents", [])

        assistant_message = {"role": "assistant", "content": answer}
        if documents:
            assistant_message["documents"] = documents
        st.session_state.messages.append(assistant_message)

    except requests.exceptions.RequestException as e:
        st.error(f"백엔드 연결 오류: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"오류: 백엔드 연결에 실패했습니다. {e}"})
    except Exception as e:
        st.error(f"예상치 못한 오류 발생: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"오류: 예상치 못한 오류가 발생했습니다. {e}"})
    
    st.rerun()
