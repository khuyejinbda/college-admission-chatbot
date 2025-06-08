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
        "search_service"
    ] = Field(
        description="Select one of the tools: search_policy, search_subject, search_admission, search_books, or search_service based on the user's question."
    )

# 구조화된 출력을 위한 LLM 설정
structured_llm = llm.with_structured_output(ToolSelector)

# 라우팅을 위한 프롬프트 템플릿
system = dedent("""You are a high school curriculum chatbot that classifies user questions into one of five categories.

Use the following routing rules:

- If the question is about how the 고교학점제 is operated—such as graduation requirements, subject completion standards, school-level implementation, course registration, or the 성취평가제—use the **search_policy** tool.

- If the question is about specific school subjects—such as what is taught in each subject, subject selection criteria, subject classification (예: 일반 선택, 진로 선택), or recommended subjects based on career interests—use the **search_subject** tool.

- If the question asks about university admissions or majors—such as what departments exist, what a certain major is about or teaches, what "계열" (academic tracks) are available, how entrance exams work, or how to prepare for 전형 types like 수시, 정시, 학과 and 학종—use the **search_admission** tool. For example, questions like “What do you learn in 호텔경영학과?” or “Tell me about 서울대 경영학과” belong here.
               and if the user asks about a university, such as in "신한대학교에 대해 알려줘", extract only the key term immediately preceding "대학교" or "대" (e.g., "신한") and use it as the core search keyword.

- If the question asks for book recommendations or summaries related to specific majors, subjects, or academic interests, use the **search_book** tool.

- If the question is about how to use the chatbot or about issues related to the MyFolio service, use the **search_service** tool.

- If the user's input includes keywords such as "세특", "수행평가", "주제", or phrases asking "what should I do", assume they are asking for a topic suggestion. In those cases, respond only with the link: https://myfolio.im/seteuk

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
    "search_service": tools.search_service,
}

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

url = "https://raw.githubusercontent.com/bdajiny/slang-dictionary/refs/heads/main/slang_dict.json"
response = requests.get(url)
raw_text = response.text
slang_dict = json.loads(raw_text) # JSON 문자열을 파싱하여 딕셔너리로 변환!

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
        if isinstance(output, list):
            all_pages = [doc.page_content for doc in output]
            if all("관련 정보를 찾을 수 없습니다" in page for page in all_pages):
                return {**state, "next_node": "llm_fallback"}
        return {**state, "next_node": datasource, "output": output}
    except Exception as e:
        print(f"Error in routing: {str(e)}")
        return {**state, "next_node": "llm_fallback"}