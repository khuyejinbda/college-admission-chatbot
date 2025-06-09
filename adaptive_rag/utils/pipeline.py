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
        tools.search_service,
        tools.search_subject
    ]
    
    return tools_list

# 툴 설정
tools = set_tools()

# AdaptiveRAG 파이프라인 빌드 함수
def build_adaptive_rag() -> StateGraph:
    """StateGraph를 빌드하고 컴파일한 adaptive_rag 객체를 반환."""
    builder = StateGraph(AdaptiveRagState)
    builder.set_entry_point("profanity_prevention")

    builder.add_node("profanity_prevention", partial(safeguard.profanity_prevention))
    builder.add_node("route_question_adaptive", router.route_question_adaptive)

    builder.add_node("search_policy", search.search_policy_adaptive)
    builder.add_node("search_subject", search.search_subject_adaptive)
    builder.add_node("search_admission", search.search_admission_adaptive)
    builder.add_node("search_book", search.search_book_adaptive)
    builder.add_node("search_service", search.search_service_adaptive)

    builder.add_node("generate", generate.generate_adaptive)
    builder.add_node("llm_fallback", generate.llm_fallback_adaptive)

    builder.add_conditional_edges(
        "profanity_prevention",
        safeguard.check_profanity_result,
        {"__end__": "__end__", "route_question_adaptive": "route_question_adaptive"}
    )

    builder.add_conditional_edges(
        "route_question_adaptive",
        lambda state: state["next_node"],
        {
            "search_policy": "search_policy",
            "search_subject": "search_subject",
            "search_admission": "search_admission",
            "search_book": "search_book",
            "search_service": "search_service",
            "llm_fallback": "llm_fallback",
        }
    )

    for node in ["search_policy","search_subject","search_admission","search_book","search_service"]:
        builder.add_edge(node, "generate")
    builder.add_edge("llm_fallback", "__end__")
    builder.add_edge("generate", "__end__")

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
