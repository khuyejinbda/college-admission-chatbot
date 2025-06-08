# adaptive_rag/state.py
from typing import TypedDict, List
from langchain_core.documents import Document

class AdaptiveRagState(TypedDict, total=False):
    question: str
    documents: List[Document]
    generation: str
    category: str
    user_id: str