"""
generate.py

이 모듈은 Adaptive RAG 챗봇에서 사용자 질문에 대한 응답을 생성하는 기능을 제공합니다. 
RAG 기반 응답(`generate_adaptive`)과 fallback 응답(`llm_fallback_adaptive`) 두 가지 전략을 지원합니다.

- `generate_adaptive`: 검색된 문서 및 대화 이력을 기반으로 사용자 질문에 응답을 생성합니다.
- `llm_fallback_adaptive`: 문서 기반 응답이 불가능하거나 relevance 판단에서 탈락한 경우, fallback 프롬프트를 활용해 응답을 생성합니다.

두 함수 모두 사용자 메모리와 MongoDB 로그 저장 기능이 포함되어 있어, 대화 흐름 유지 및 사용성 분석이 가능합니다.
"""

from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  
from adaptive_rag.utils.memory import get_user_memory
from adaptive_rag.utils.mongoDB import save_chat_log
from langchain_openai import ChatOpenAI
from pprint import pprint
from dotenv import load_dotenv
import os
from adaptive_rag.utils.state import AdaptiveRagState
from adaptive_rag.utils.prompts import get_prompt_by_key

# API 키 정보 로드
load_dotenv()

# API 키 읽어오기
openai_api_key = os.environ.get('OPENAI_API_KEY')

# 기본 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

def generate_adaptive(state: AdaptiveRagState):
    """
    문서 기반 RAG 응답 생성 함수.

    검색된 문서와 대화 이력을 바탕으로 LLM이 질문에 응답을 생성합니다.
    응답은 유저 메모리에 저장되고, MongoDB에도 로그 형태로 저장됩니다.

    Args:
        state (AdaptiveRagState): 질문, 문서, 사용자 ID 등이 담긴 상태 객체

    Returns:
        dict: 상태에 'generation' 키가 추가된 딕셔너리
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    user_id = state.get("user_id", "anonymous")
    category = state.get("category", "미지정")
    prompt_key = state.get("prompt_key", None)
    
    # 프롬프트 키에 해당하는 템플릿 불러오기
    prompt_template = get_prompt_by_key(prompt_key)
 
    # 프롬프트 템플릿이 없으면 에러 메시지 반환
    if not prompt_template:
      return {"generation": "적절한 프롬프트를 찾을 수 없습니다."}

    # 유저 메모리 가져오기
    memory = get_user_memory(user_id)

    # 문서 리스트 형태로 정제
    if not isinstance(documents, list):
        documents = [documents]

    # 문서 내용을 텍스트 형태로 병합
    documents_text = "\n\n".join([
        f"---\n본문: {doc.page_content}\n메타데이터:{str(doc.metadata)}\n---"
        for doc in documents
    ])

    # 이전 대화 이력 가져오기
    history = memory.chat_memory.messages  # 이전 대화 리스트
    history_text = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"Bot: {m.content}"
        for m in history
    ])

    # RAG 체인 실행: prompt → LLM → 출력 파서
    rag_chain = prompt_template | llm | StrOutputParser()
    generation = rag_chain.invoke({
        "documents": documents_text,
        "question": question,
        "history": history_text
    })

    # 메모리 & 로그 저장
    memory.chat_memory.add_user_message(HumanMessage(content=question))
    memory.chat_memory.add_ai_message(AIMessage(content=generation))

    save_chat_log(question, generation, user_id=user_id, category=category)

    return {**state, "generation": generation}


def llm_fallback_adaptive(state: AdaptiveRagState):
    """
    관련 문서가 없거나 fallback 경로로 전환된 질문에 대한 응답 생성 함수.

    문서를 사용하지 않고 fallback 전용 프롬프트만으로 응답을 생성합니다.
    응답은 유저 메모리 및 MongoDB에 저장됩니다.

    Args:
        state (AdaptiveRagState): 질문, 사용자 ID 등이 담긴 상태 객체

    Returns:
        dict: 상태에 'generation' 키가 추가된 딕셔너리
    """
    question = state.get("question", "")
    user_id = state.get("user_id", "anonymous")
    category = state.get("category", "미지정")

    # fallback 프롬프트 로드
    prompt_template = get_prompt_by_key("fallback")

    memory = get_user_memory(user_id)
    
    # fallback 체인 실행
    llm_chain = prompt_template | llm | StrOutputParser()
    generation = llm_chain.invoke({"question": question})

    # 메모리에 저장
    memory.chat_memory.add_user_message(HumanMessage(content=question))
    memory.chat_memory.add_ai_message(AIMessage(content=generation))

    # 로그 저장
    save_chat_log(question, generation, user_id=user_id, category=category)

    return {**state, "generation": generation}