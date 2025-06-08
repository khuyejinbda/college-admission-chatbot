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
    history = memory.chat_memory.messages[-3:]  # 최근 5개만
    history_text = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"Bot: {m.content}"
        for m in history
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "아래는 유저와의 대화 기록이야. 최근 질문을 더 구체적으로 바꿔줘."),
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
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 고교학점제 운영 정보를 찾을 수 없습니다.")]}


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
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 과목 정보를 찾을 수 없습니다.")]}

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
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 입시 정보를 찾을 수 없습니다.")]}

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
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 도서 정보를 찾을 수 없습니다.")]}


def search_service_adaptive(state: AdaptiveRagState):
    """
    Node for searching the 베어러블 service information
    """
    question = state["question"]
    user_id = state.get("user_id", "anonymous")
    memory = get_user_memory(user_id)

    # 질문 리프레이징
    enriched_question = rephrase_question_with_history(memory, question)

    docs = tools.search_service.invoke(enriched_question)
    if len(docs) > 0:
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 베어러블 서비스 정보를 찾을 수 없습니다.")]}