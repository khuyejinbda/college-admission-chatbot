"""
router.py

이 모듈은 Adaptive RAG 기반 챗봇에서 사용자 질문을 분석하여 적절한 검색 도구(tool)로 라우팅하는 역할을 수행합니다.
LangChain의 structured output 기능과 프롬프트를 활용하여 질문 유형에 따라 policy/subject/admission/book/seteuk/fallback 중 하나를 선택합니다.

핵심 기능:
- slang 치환을 통한 질문 전처리
- 질문 분류 후 해당 search tool 실행
- 이전에 시도한 도구를 제외한 재라우팅 수행
- 선택된 도구 결과를 state에 `output`, `next_node`, `prompt_key` 등의 정보로 추가

사용 도구 목록:
- search_policy
- search_subject
- search_admission
- search_book
- search_seteuk
- llm_fallback
"""

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from textwrap import dedent
from langchain_openai import ChatOpenAI
from pprint import pprint
from dotenv import load_dotenv
import os
from adaptive_rag.utils.state import AdaptiveRagState
from adaptive_rag.utils import tools, slang
import re
import json
import requests
from adaptive_rag.utils.check import check_relevance

# API 키 정보 로드
load_dotenv()

# API 키 읽어오기
openai_api_key = os.environ.get('OPENAI_API_KEY')

# 기본 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# 라우팅 결정용 데이터 모델
class ToolSelector(BaseModel):
    """Routes the user question to the most appropriate tool."""
    tool: Literal[
        "search_policy", 
        "search_subject", 
        "search_admission", 
        "search_book", 
        "search_seteuk",
        "llm_fallback"  # Fallback option if no tool is suitable
    ] = Field(
        description="Select one of the tools: search_policy, search_subject, search_admission, search_books, or search_seteuk, llm_fallback based on the user's question."
    )

# 구조화된 출력을 위한 LLM 설정
structured_llm = llm.with_structured_output(ToolSelector)

# 라우팅을 위한 프롬프트 템플릿
system = dedent("""You are a high school curriculum chatbot that classifies user questions into one of six categories.

Use the following routing rules:

- If the question is about how the 고교학점제 is operated—such as graduation requirements, subject completion standards, school-level implementation, course registration, or the 성취평가제—use the **search_policy** tool.

- If it's a question about a particular school subject  
  The content taught in each subject, subject selection criteria, subject classification, subject-related activities and 세특(세부특기 능력사항),  
  (e.g., general choice, career choice), recommended subject based on career interest,  
  Alternatively, the **subject grade calculation method (e.g., grading system, achievement)** — use the **search_subject** tool.  
  👉 Use this tool **when the user is asking about the subject itself**: what the subject teaches, who should take it, or how it's evaluated.

- If the question asks about university admissions or majors—such as what departments exist, what a certain major is about or teaches, what "계열" (academic tracks) are available, how entrance exams work, or how to prepare for 전형 types like 수시, 정시, 학과 and 학종—use the **search_admission** tool.  
  For example, questions like “What do you learn in 호텔경영학과?” or “Tell me about 서울대 경영학과” belong here.  
  If the user asks about a university, such as in "신한대학교에 대해 알려줘", extract only the key term immediately preceding "대학교" or "대" (e.g., "신한") and use it as the core search keyword.

- If the question asks for book recommendations or summaries related to specific majors, subjects, or academic interests, use the **search_book** tool.

- If the question asks for 세특 (세부능력 및 특기사항) topic suggestions—especially topic ideas, example activities, or related keywords—use the **search_seteuk** tool.  
  👉 Use this tool **when the user is asking about what kind of 세특 activity or topic to write** related to a subject or major, such as:  
  “세특 활동 추천해줘”, “어떤 탐구 주제를 쓰면 좋을까?”, “~~학과랑 연계된 수학 세특 주제가 궁금해”, "~~랑 관련된 주제 추천해줘".

🟨 To clarify:
Use the following rules to distinguish between 세특-related questions:

1. If the question is about the **subject itself**, such as:  
   - 과목의 개념이 궁금해요  
   - 이 과목에서 뭘 배우나요?  
   - 성취수준이 어떻게 되나요?  
   → Use **search_subject**

2. If the question is about **세특 topics related to a subject or major**, such as:  
   - 수학이랑 관련된 세특 주제가 궁금해요  
   - 경영학과에 맞는 세특 활동 추천해줘  
   - 어떤 탐구 주제를 쓰면 좋을까?  
   → Use **search_seteuk**

3. If the question is about the **definition, rules, or writing method of 세특 itself**, such as:  
   - 세특이 뭔가요?  
   - 세특은 어떻게 작성하나요?  
   - 생활기록부에는 세특을 어떻게 기재해요?  
   → Use **search_policy**


- 맥락을 고려하여서, 고등학교 과목에 대한 질문이라면 search_subject 툴을 사용하고, 대학 전공에 대한 질문이라면 search_admission 툴을 사용합니다.

Always choose the single most relevant tool that best matches the user's intent.
""")


route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 질문 라우터 정의
question_router = route_prompt | structured_llm

# 너의 툴 이름과 함수 맵핑
tool_map = {
    "search_policy": tools.search_policy,
    "search_subject": tools.search_subject,
    "search_admission": tools.search_admission,
    "search_book": tools.search_book,
    "search_seteuk": tools.search_seteuk,
}

# 툴 실행 후 결과를 가져오는 함수
def run_tool_and_get_output(question: str) -> dict:
    # 툴 선택 (output은 안 나옴)
    result = question_router.invoke({"question": question})
    tool_name = result.tool

    # 툴 실행 (직접 실행해야 함)
    if tool_name in tool_map:
        output = tool_map[tool_name](question)  # 여기에 실제 검색 결과가 담김
    else:
        output = None

    return {
        "tool": tool_name,
        "output": output
    }

# 슬랭 사전 로드 (여기서 직접 처리)
url = "https://raw.githubusercontent.com/khuyejinbda/college-admission-chatbot/main/slang_dict.json"
response = requests.get(url)
raw_text = response.text
slang_dict = json.loads(raw_text) # JSON 문자열을 파싱하여 딕셔너리로 변환!

# 라우팅 함수 정의
def route_question_adaptive(state: AdaptiveRagState) -> AdaptiveRagState:
    # 1) 슬랭 전처리 (여기서 직접 처리)
    q = state["question"]
    if any(s in q for s in slang_dict.keys()):
        # replace_slang_word 가 {"question": "..."} 를 리턴하므로 
        state["question"] = slang.replace_slang_word(q, slang_dict)["question"]

    # 2) 기존 라우팅 로직
    try:
        result = run_tool_and_get_output(state["question"])
        datasource = result["tool"]
        output = result["output"]
        return {**state, "next_node": datasource, "output": output, "prompt_key": datasource.replace("search_", ""), "visited_nodes": [datasource], "retried": False}
    except Exception as e:
        print(f"Error in routing: {str(e)}")
        return {**state, "next_node": "llm_fallback", "prompt_key": "fallback"}

# 재라우팅을 위한 프롬프트 빌드 함수
def build_re_route_prompt(visited: list[str]):
    visited_str = ", ".join(visited) if visited else "없음"
    routing_hint = (
        f"🔁 현재까지 시도한 도구 목록: {visited_str}\n"
        f"이번에는 **이전에 시도하지 않은 도구 중에서 반드시 하나만** 선택해야 해.\n"
        f"⚠️ 절대 이전과 동일한 도구를 반복해서 선택하지 마.\n"
        f"만약 적절한 도구가 없다고 판단되면 'llm_fallback'을 선택해.\n"
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", routing_hint+ system ),
            ("human", "{question}"),
        ]
    )

# 재라우팅 함수 정의
def re_route_question_adaptive(state: AdaptiveRagState) -> AdaptiveRagState:
    question = state["question"]
    visited = state.get("visited_nodes", [])

    # 슬랭 정제
    if any(s in question for s in slang_dict.keys()):
        state["question"] = slang.replace_slang_word(question, slang_dict)["question"]
        question = state["question"]

    try:
        # visited-aware prompt 구성
        prompt = build_re_route_prompt(visited)
        rerouter = prompt | structured_llm
        result = rerouter.invoke({"question": question})
        tool_name = result.tool

        # 툴 중복 방지
        if tool_name in visited:
            print(f"[RE_ROUTE] LLM이 같은 노드({tool_name})를 반환하여 fallback.")
            new_state = {**state, "visited_nodes": visited + [tool_name]}
            return new_state

        # 툴 실행
        if tool_name in tool_map:
            output = tool_map[tool_name](question)
        else:
            output = None

        return {
        **state,
        "next_node": tool_name,
        "output": output,
        "visited_nodes": visited + [tool_name],
        "retried": True,  # ✅ 재시도 플래그 갱신
        "prompt_key": tool_name.replace("search_", "")  # 예: "policy"
        }

    except Exception as e:
        print(f"[RE_ROUTE ERROR] {str(e)}")
        return {**state, "next_node": "llm_fallback", "prompt_key": "fallback"}
