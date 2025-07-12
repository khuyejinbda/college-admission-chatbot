from adaptive_rag.utils import tools, safeguard, search, generate, memory, mongoDB, router, slang, state
from adaptive_rag.utils.state import AdaptiveRagState

from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from functools import partial

import random
import threading
import sys
from typing import Dict, Any, Union # Added Union

# 툴 설정 함수
def set_tools():
    """
    Set up the tools for the Adaptive RAG pipeline.
    """
    tools_list = [
        tools.search_policy,
        tools.search_admission,
        tools.search_book,
        tools.search_subject,
        tools.search_seteuk
    ]
    
    return tools_list

# 툴 설정
tools = set_tools()

def build_adaptive_rag() -> StateGraph:
    """
    LangGraph 기반 Adaptive RAG 챗봇을 위한 상태 그래프 생성 함수.
    각 노드들은 질문 처리의 단계(욕설 필터링 → 라우팅 → 검색 → 관련성 평가 → 생성)로 구성됨.

    Returns:
        StateGraph: 실행 가능한 상태 그래프

    - 구조
    [profanity_prevention] 
    → (clean) → [route_question_adaptive] 
        → [search_xxx] 
            → [check_relevance]
                → (1) [generate] → end
                → (0, not retried) → [re_route_question_adaptive] → [search_xxx2] → ...
                → (0, retried) → [llm_fallback] → end
    → (profane) → end

    """

    # 그래프 초기화 (state 타입은 AdaptiveRagState)
    builder = StateGraph(AdaptiveRagState)

    # 시작 지점을 욕설 필터링 노드로 지정
    builder.set_entry_point("profanity_prevention")

    # === 주요 노드 등록 ===

    # 1. 욕설 필터링 (욕설 감지 및 종료/계속 판단)
    builder.add_node("profanity_prevention", partial(safeguard.profanity_prevention))

    # 2. 라우팅 (질문 유형에 따라 search 노드 결정)
    builder.add_node("route_question_adaptive", router.route_question_adaptive)
    builder.add_node("re_route_question_adaptive", router.re_route_question_adaptive)

    # 3. 관련성 판단 (검색 결과와 질문이 연결되는지 판단)
    builder.add_node("check_relevance", check.check_relevance)

    # 4. 검색 노드 (주제별로 분리)
    builder.add_node("search_policy", search.search_policy_adaptive)
    builder.add_node("search_subject", search.search_subject_adaptive)
    builder.add_node("search_admission", search.search_admission_adaptive)
    builder.add_node("search_book", search.search_book_adaptive)
    builder.add_node("search_seteuk", search.search_seteuk_adaptive)

    # 5. 응답 생성 또는 fallback
    builder.add_node("generate", generate.generate_adaptive)
    builder.add_node("llm_fallback", generate.llm_fallback_adaptive)

    # === 상태 간 연결 정의 ===

    # Step 1: 욕설이 감지되면 종료, 없으면 다음 단계로 진행
    builder.add_conditional_edges(
        "profanity_prevention",
        safeguard.check_profanity_result,
        {
            "__end__": "__end__",  # 욕설 감지 시 종료
            "route_question_adaptive": "route_question_adaptive",  # 정상 질문 → 라우팅
        }
    )

    # Step 2: 라우팅 결과에 따라 적절한 검색 노드로 이동
    builder.add_conditional_edges(
        "route_question_adaptive",
        lambda state: state["next_node"],
        {
            "search_policy": "search_policy",
            "search_subject": "search_subject",
            "search_admission": "search_admission",
            "search_book": "search_book",
            "search_seteuk": "search_seteuk",
            "llm_fallback": "llm_fallback",
        }
    )

    # Step 3: 모든 search 노드의 결과는 check_relevance로 이동
    for node in ["search_policy", "search_subject", "search_admission", "search_book", "search_seteuk"]:
        builder.add_edge(node, "check_relevance")

    # Step 4: 관련성이 높으면 generate로, 낮고 재시도 가능하면 재라우팅, 아니면 fallback
    builder.add_conditional_edges(
        "check_relevance",
        lambda state: (
            "generate" if state.get("relevance_score") == 1
            else (
                "re_route_question_adaptive" if not state.get("retried", False)
                else "llm_fallback"
            )
        ),
        {
            "generate": "generate",
            "re_route_question_adaptive": "re_route_question_adaptive",
            "llm_fallback": "llm_fallback",
        }
    )

    # Step 5: 재라우팅 결과에 따라 새로운 노드로 이동
    builder.add_conditional_edges(
        "re_route_question_adaptive",
        lambda state: (
            "llm_fallback"  # 재시도인데 또 같은 노드라면 fallback
            if state["next_node"] in state.get("visited_nodes", [])
            else state["next_node"]  # 새 노드이면 해당 노드로
        ),
        {
            "search_policy": "search_policy",
            "search_subject": "search_subject",
            "search_admission": "search_admission",
            "search_book": "search_book",
            "search_seteuk": "search_seteuk",
            "llm_fallback": "llm_fallback",
        }
    )

    # Step 6: generate 또는 fallback 이후 종료
    builder.add_edge("generate", "__end__")
    builder.add_edge("llm_fallback", "__end__")

    # 그래프 최종 컴파일
    return builder.compile()

# Global graph instance for API usage
compiled_graph_instance: Union[StateGraph, None] = None

def initialize_graph_for_api():
    """Initializes the RAG graph for API usage if not already initialized."""
    global compiled_graph_instance
    if compiled_graph_instance is None:
        print("Initializing RAG graph for API...")
        compiled_graph_instance = build_adaptive_rag()
        print("RAG graph initialized for API.")

def get_chatbot_response(question: str, user_id: str, category :str) -> Dict[str, Any]:
    """
    Processes a question using the RAG graph and returns the chatbot's response.
    Designed for API usage.
    """
    global compiled_graph_instance
    if compiled_graph_instance is None:
        # This should ideally be called by the FastAPI app startup,
        # but initialize here if accessed directly before app startup.
        initialize_graph_for_api()
    
    if compiled_graph_instance is None: # Still None after trying to initialize
        return {"error": "Graph could not be initialized."}

    inputs = {
        "question": question,
        "user_id": user_id,
        "category": category
    }
    final_node_output_state = {}
    
    # The stream yields dictionaries where keys are node names and values are the state dicts.
    # We want the state from the node that produces the 'generation'.
    for output_chunk in compiled_graph_instance.stream(inputs):
        for node_name, state_after_node in output_chunk.items():
            # We are interested in the state that contains the 'generation'.
            # This will typically be the state after 'generate' or 'llm_fallback' nodes.
            # The last such state before the stream ends should be the overall final state.
            final_node_output_state = state_after_node 

    if 'generation' not in final_node_output_state:
        # Check if it's an error or if the graph ended via a path that doesn't set 'generation'.
        # For now, we assume 'generate' or 'llm_fallback' always sets 'generation'.
        print(f"Warning: 'generation' key missing. Last state: {final_node_output_state}")
        return {"error": "Generation not found in the final state.", "details": final_node_output_state}

    return final_node_output_state

def run_chatbot():
    """
    챗봇 실행 (5분 무응답 시 자동 종료)
    """
    graph = build_adaptive_rag()
    user_id = random.randint(1, 1_000_000)

    def timeout_exit():
        print("\n5분 동안 입력이 없어 챗봇을 종료합니다.")
        sys.exit(0)

    # 최초 타이머 설정
    timer = threading.Timer(300, timeout_exit)
    timer.start()

    try:
        while True:
            question = input("질문을 입력해주세요 > ").strip()
            # 입력이 들어오면 기존 타이머 취소 후 재시작
            timer.cancel()
            if not question:
                print("종료합니다.")
                break

            inputs = {
                "question": question,
                "user_id": user_id,
                "category": None  # 일단 로컬에서는 없음
            }

            for output in graph.stream(inputs):
                for key, value in output.items():
                    final_output = value

            print(f"🤖 답변: {final_output['generation']}")

            # 다시 5분 타이머 시작
            timer = threading.Timer(300, timeout_exit)
            timer.start()

    finally:
        # 프로그램 종료 전 타이머는 반드시 취소
        timer.cancel()

# 챗봇 실행
if __name__ == "__main__":
    run_chatbot()
