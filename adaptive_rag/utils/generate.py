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

# API 키 정보 로드
load_dotenv()

# API 키 읽어오기
openai_api_key = os.environ.get('OPENAI_API_KEY')

# 기본 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

def generate_adaptive(state: AdaptiveRagState):
    question = state.get("question", "")
    documents = state.get("documents", [])
    user_id = state.get("user_id", "anonymous")

    # 유저 메모리 가져오기
    memory = get_user_memory(user_id)

    if not isinstance(documents, list):
        documents = [documents]

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

    # RAG 프롬프트 정의
    prompt_with_context = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant answering questions based on provided documents. Follow these guidelines:

    1. Use only information from the given documents.
    2. If the document lacks relevant info, say "The provided documents don't contain information to answer this question."
    3. Cite relevant parts of the document in your answers.
    4. Don't speculate or add information not in the documents.
    5. Keep answers concise and clear.
    6. Omit irrelevant information.
    7. 관련된 문서가 없을 경우, 
        "The provided documents don't contain information to answer this question."라고 답변해줘.
    8. 이모티콘 넣어서 다정하게 답변해줘
    9.If the user's input includes keywords such as "세특", "수행평가", "주제", or phrases asking "what should I do", assume they are asking for a topic suggestion. In those cases, respond only with the link: https://myfolio.im/seteuk
         """

    ),
        ("human", "Answer the following question using these documents:\n\n[Documents]\n{documents}\n\n[Question]\n{question}"),
    ])

    rag_chain = prompt_with_context | llm | StrOutputParser()
    generation = rag_chain.invoke({
        "documents": documents_text,
        "question": question                                    
    })

    # 메모리 & 로그 저장
    memory.chat_memory.add_user_message(HumanMessage(content=question))
    memory.chat_memory.add_ai_message(AIMessage(content=generation))

    save_chat_log(question, generation, user_id=user_id)

    return {"generation": generation}


from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  

def llm_fallback_adaptive(state: AdaptiveRagState):
    question = state.get("question", "")
    user_id = state.get("user_id", "anonymous")

    # 유저별 memory 가져오기
    memory = get_user_memory(user_id)

    # 이전 대화 context 구성
    history = memory.chat_memory.messages
    history_text = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"Bot: {m.content}"
        for m in history
    ])

    # LLM Fallback 프롬프트 정의
    prompt_with_context = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant helping with various topics. Follow these guidelines:

    There are two possible situations:

    1. If the {question} is relevant to topics like school policies, curriculum, admissions, book, or services 등 학교에 관련된 정보,
        respond by clearly stating: "관련된 문서를 찾을 수 없습니다."

    2. If the question is unrelated to those topics (e.g., public holidays, general culture, history, daily life),
       simply answer it using your general knowledge.

    In all cases:
    - Provide accurate and helpful information to the best of your ability.
    - Express uncertainty when unsure; avoid speculation.
    - Keep answers concise yet informative.
    - Inform users they can ask for clarification if needed.
    - Respond ethically and constructively.
    - Mention reliable general sources when applicable if needed.
    - If the user's input includes keywords such as "세특", "수행평가", "주제", or phrases asking "what should I do", assume they are asking for a topic suggestion. In those cases, respond only with the link: https://myfolio.im/seteuk

    """),
        ("human", "{question}"),
    ])

    llm_chain = prompt_with_context | llm | StrOutputParser()
    generation = llm_chain.invoke({"question": question})

    # 메모리에 저장
    memory.chat_memory.add_user_message(HumanMessage(content=question))
    memory.chat_memory.add_ai_message(AIMessage(content=generation))

    # 로그 저장
    save_chat_log(question, generation, user_id=user_id)

    return {"generation": generation}