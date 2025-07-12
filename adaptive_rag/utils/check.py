"""
check.py

이 모듈은 Adaptive RAG 프레임워크에서 사용자 질문과 검색된 문서 사이의 의미적 관련성을 평가하는 기능을 제공합니다. 
LangChain의 ChatPromptTemplate과 OpenAI LLM을 활용하여, 질문-문서 간 관련 여부를 이진 분류(0 또는 1)로 판단하며, 
해당 결과는 RAG 시스템에서 재검색 여부(re-route) 또는 fallback 처리를 결정하는 데 사용됩니다.

주요 기능:
- 질문과 문서 간 의미적 관련성 평가
- LLM을 활용한 프롬프트 기반 분류
- 판단 결과(0 또는 1)에 따라 relevance_score 및 prompt_key 업데이트
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from textwrap import dedent
from langchain_openai import ChatOpenAI
from pprint import pprint
from adaptive_rag.utils.state import AdaptiveRagState

# .env 파일에서 환경변수 불러오기
load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

# OpenAI LLM 모델 설정 (gpt-4o-mini 사용)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# 관련성 판단을 위한 프롬프트 설정
check_prompt = ChatPromptTemplate.from_messages([
    ("system", 
    "너는 고등학생의 질문에 답변하는 교육 챗봇을 위한 RAG 평가 전문가야.\n"
    "사용자의 질문과 아래 문서들이 **논리적으로 연결되어 답변이 가능한지**를 판단해.\n\n"
    "문서가 학생의 질문에 의미 있는 정보를 제공할 수 있어야 해.\n"
    "너는 논리적인 판단을 하는 전문가이므로, 철저하게 **학생의 질문에 문서가 응답 가능할지를 기준으로** 판단해야 해.\n\n"
     "사용자의 질문과 아래 문서들이 **의미적으로 연결되어 있고**, 문서가 질문에 답할 수 있는 실질적 정보를 담고 있는지를 판단해.\n\n"
     "질문과 문서의 단어가 정확히 일치하지 않더라도, 같은 주제를 다루거나 관련 맥락이 담겨 있다면 '1'을 줘.\n"
     "반대로 전혀 다른 주제거나, 질문의 의도와 무관한 정보만 있다면 '0'을 줘.\n\n"
     "단, **가능한 한 유연하게 판단**하되, 문서가 질문에 대해 의미 있는 힌트나 정보, 규정을 제공하지 않으면 '0'으로 판단해야 해.\n\n"
    "출력은 반드시 숫자 '1' 또는 '0' 중 하나로만 해야 해.\n"
    "절대 설명이나 다른 텍스트를 추가하지 마.\n"
    "무조건 1 또는 0만 단독으로 출력해야 해."),
    ("human", 
     "질문: {question}\n\n문서 미리보기:\n{docs}\n\n관련 여부만 숫자로 답해:")
])

# 프롬프트, 모델, 출력 파서를 연결한 평가 체인 생성
llm_check_chain = check_prompt | llm | StrOutputParser()

def check_relevance(state: AdaptiveRagState) -> AdaptiveRagState:
    """
    사용자 질문과 검색된 문서 간의 의미적 관련성을 평가하여,
    AdaptiveRagState에 relevance_score(0 또는 1)를 추가한다.
    
    관련성이 없는 경우 prompt_key를 'fallback'으로 설정한다.
    
    Args:
        state (AdaptiveRagState): 현재 질문, 문서 등 정보를 포함한 상태 객체

    Returns:
        AdaptiveRagState: 관련성 점수와 prompt_key가 추가된 상태
    """
    docs = state.get("documents", [])
    question = state.get("question")

    # 문서가 list인 경우, 각 문서에서 최대 1000자씩 추출하여 미리보기 구성
    docs_preview = "\n\n".join(doc.page_content[:1000] for doc in docs) if isinstance(docs, list) else ""

    try:
        # LLM을 통해 관련성 판단 ('1' 또는 '0')
        decision = llm_check_chain.invoke({
            "question": question,
            "docs": docs_preview
        }).strip()

        # 예외적인 출력 방지: '1' 또는 '0'이 아닌 경우 fallback 처리
        if decision not in ("0", "1"):
            print(f"[CHECK WARNING] Unexpected output: {decision}")
            score = 0
        else:
            score = int(decision)

    except Exception as e:
        # 오류 발생 시 기본값 '0'으로 처리
        print(f"[CHECK ERROR] {str(e)}")
        score = 0

    # relevance_score와 prompt_key 업데이트
    updated_state = {**state, "relevance_score": score}
    if score == 0:
        updated_state["prompt_key"] = "fallback"

    return updated_state


