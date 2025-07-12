from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from adaptive_rag.utils.state import AdaptiveRagState
from adaptive_rag.utils import tools
from adaptive_rag.utils.memory import get_user_memory
from langchain_core.prompts import ChatPromptTemplate

from langchain.memory import ConversationBufferWindowMemory
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  

# API 키 정보 로드
load_dotenv()

# API 키 읽어오기
openai_api_key = os.environ.get('OPENAI_API_KEY')

# 기본 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

def rephrase_question_with_history(memory, current_question):
    history = memory.chat_memory.messages[-5:]  # 최근 5개만
    history_text = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"Bot: {m.content}"
        for m in history
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 고도의 대화 이해 및 재작성 전문가입니다.아래 대화 이력을 꼼꼼히 분석하여, 사용자의 원래 의도를 온전히 반영하면서 핵심 정보를 보강하고 자연스럽고 간결한 질문으로 재작성해 주세요."),
        ("human", f"대화 기록:\n{history_text}\n\n질문: {current_question}\n\n보완된 질문:")
    ])

    rephrase_chain = prompt | llm | StrOutputParser()
    enriched = rephrase_chain.invoke({"question": current_question})
    return enriched

def search_policy_adaptive(state: AdaptiveRagState):
    """
    Node for searching information in the 고교학점제 운영
    """
    question = state["question"]
    user_id = state.get("user_id", "anonymous")
    memory = get_user_memory(user_id)

    # 질문 리프레이징
    enriched_question = rephrase_question_with_history(memory, question)

    docs = tools.search_policy.invoke(enriched_question)
    if len(docs) > 0:
        return {**state, "documents": docs}
    else:
        return {**state, "documents": [Document(page_content="관련 정보를 찾을 수 없습니다")], "prompt_key": "fallback"}


def search_subject_adaptive(state: AdaptiveRagState):
    """
    Node for searching information in the subject whthin the 고교학점제
    """
    question = state["question"]
    user_id = state.get("user_id", "anonymous")
    memory = get_user_memory(user_id)

    # 질문 리프레이징
    enriched_question = rephrase_question_with_history(memory, question)

    docs = tools.search_subject.invoke(enriched_question)

    if len(docs) > 0:
        return {**state, "documents": docs}
    else:
        return {**state, "documents": [Document(page_content="관련 정보를 찾을 수 없습니다")], "prompt_key": "fallback"}

def search_admission_adaptive(state: AdaptiveRagState):
    """
    Node for searching information in the admission
    """
    question = state["question"]
    user_id = state.get("user_id", "anonymous")
    memory = get_user_memory(user_id)

    # 질문 리프레이징
    enriched_question = rephrase_question_with_history(memory, question)

    docs = tools.search_admission.invoke(enriched_question)
    if len(docs) > 0:
        return {**state, "documents": docs}
    else:
        return {**state, "documents": [Document(page_content="관련 정보를 찾을 수 없습니다")], "prompt_key": "fallback"}

def search_book_adaptive(state: AdaptiveRagState):
    """
    Node for searching information in the book
    """
    question = state["question"]
    user_id = state.get("user_id", "anonymous")
    memory = get_user_memory(user_id)

    # 질문 리프레이징
    enriched_question = rephrase_question_with_history(memory, question)
    
    docs = tools.search_book.invoke(enriched_question)
    if len(docs) > 0:
        return {**state, "documents": docs}
    else:
        return {**state, "documents": [Document(page_content="관련 정보를 찾을 수 없습니다")], "prompt_key": "fallback"}


def search_seteuk_adaptive(state: AdaptiveRagState):
    """
    Node for searching the 세특 추천 관련 information
    """
    question = state["question"]
    user_id = state.get("user_id", "anonymous")
    memory = get_user_memory(user_id)

    # 질문 리프레이징
    enriched_question = rephrase_question_with_history(memory, question)

    docs = tools.search_seteuk.invoke(enriched_question)
    if len(docs) > 0:
        return {**state, "documents": docs}
    else:
        return {**state, "documents": [Document(page_content="관련 정보를 찾을 수 없습니다")], "prompt_key": "fallback"}