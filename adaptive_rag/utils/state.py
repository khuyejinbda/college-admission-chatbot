from typing import TypedDict, List
from langchain_core.documents import Document

class AdaptiveRagState(TypedDict, total=False):
    """
    LangGraph 에이전트와 함께 사용하는 RAG 상태 객체
    - LangGraph 프레임워크의 에이전트 상태(state) 표현을 위해 설계
    - total=False 로 설정하여 모든 필드가 optional(선택적)임을 지정
    """
    question: str # 사용자의 질문
    documents: List[Document] # 검색된 문서 목록
    generation: str # 생성된 답변
    category: str
    user_id: str
    prompt_key : str # 프롬프트 키 (예: "policy", "admission" 등)
    retried: bool = False  # 재시도 여부
    visited_nodes: List[str]  # ✅ 검색에 사용된 노드 추적
    relevance_score: int = 0  # 관련성 점수 (0 또는 1)
    next_node: str # 다음 실행할 노드 이름
