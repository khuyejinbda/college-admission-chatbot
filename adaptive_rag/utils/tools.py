# 필요한 라이브러리 임포트
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from typing import List
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere

# API 키 정보 로드
load_dotenv()

# API 키 읽어오기
openai_api_key = os.environ.get('OPENAI_API_KEY')
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
cohere_api_key = os.environ.get("COHERE_API_KEY")

# OpenAI 임베딩 인스턴스 생성
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-large',
    openai_api_key=openai_api_key
)

compressor = CohereRerank(model="rerank-multilingual-v3.0",top_n=4)

# 운영 문의 정보 검색
pinecone_policy = PineconeVectorStore.from_documents(
    documents=[], # 빈 리스트로 초기화
    index_name="college-admission-chatbot",   # 인덱스 이름
    embedding=embeddings,               # 임베딩 인스턴스
    pinecone_api_key=pinecone_api_key,
    namespace="policy", # 네임스페이스 설정 : 운영 문의 -> policy
)

# 2. Pinecone 리트리버를 LangChain retriever로 감싸기
policy_retriever = pinecone_policy.as_retriever(search_kwargs={"k": 6})

compression_retriever_policy = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=policy_retriever
)

# 운영 정보 검색 tool 정의
@tool
def search_policy(query: str) -> List[Document]:
    """
    Securely search and access operational information related to the High School Credit System,
    including graduation requirements, academic regulations, subject completion standards, and general policy guidelines.

    To maintain data integrity and clarity, use this tool only for questions about system operations of the High School Credit System,
    such as curriculum rules, credit units, or graduation criteria.
    """
    docs = compression_retriever_policy.invoke(query)
    if len(docs) > 0:
        return docs
    
    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]

# 과목 정보 검색
pinecone_subject = PineconeVectorStore.from_documents(
    documents=[], # 빈 리스트로 초기화
    index_name="college-admission-chatbot",   # 인덱스 이름
    embedding=embeddings,               # 임베딩 인스턴스
    pinecone_api_key=pinecone_api_key,
    namespace="subject", # 네임스페이스 설정: 과목 문의 -> subject
)

# 2. Pinecone 리트리버를 LangChain retriever로 감싸기
subject_retriever = pinecone_subject.as_retriever(search_kwargs={"k": 6})

compression_retriever_subject = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=subject_retriever
)

# 과목 점보 검색 tool 정의
@tool
def search_subject(query: str) -> List[Document]:
    """
    Securely search and access information about subject within the High School Credit System,  
    including subject descriptions, learning objectives, curriculum content, and their relevance to specific career paths.

    To ensure appropriate guidance, use this tool only for questions related to subject within the High School Credit System
    """
    docs = compression_retriever_subject.invoke(query)
    if docs:
        return docs
    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]

admission_compressor = CohereRerank(model="rerank-multilingual-v3.0",top_n=7)

# 입시 정보 검색
pinecone_admission = PineconeVectorStore.from_documents(
    documents=[], # 빈 리스트로 초기화
    index_name="college-admission-chatbot",   # 인덱스 이름
    embedding=embeddings,               # 임베딩 인스턴스
    pinecone_api_key=pinecone_api_key,
    namespace="admission", # 네임스페이스 설정: 과목 문의 -> subject
)

# 2. Pinecone 리트리버를 LangChain retriever로 감싸기
admission_retriever = pinecone_admission.as_retriever(search_kwargs={"k": 30})

compression_retriever_admission = ContextualCompressionRetriever(
    base_compressor=admission_compressor, base_retriever=admission_retriever
)

# 입시 점보 검색 tool 정의
@tool
def search_admission(query: str) -> List[Document]:
    """
    Search and access detailed information related to college admissions under the High School Credit System,
    especially focusing on university departments and majors. This includes what each major typically teaches,
    college overviews (location, offered programs, general info), and explanations of admissions-related terminology.
    
    Use this tool for queries about selecting a major or university, understanding admission systems, or learning about specific departments.
    """

    docs = compression_retriever_admission.invoke(query)
    if len(docs) > 0:
        return docs
    
    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]

# 도서 정보 검색
pinecone_book = PineconeVectorStore.from_documents(
    documents=[], # 빈 리스트로 초기화
    index_name="college-admission-chatbot",   # 인덱스 이름
    embedding=embeddings,               # 임베딩 인스턴스
    pinecone_api_key=pinecone_api_key,
    namespace="book", # 네임스페이스 설정: 과목 문의 -> subject
)

# 2. Pinecone 리트리버를 LangChain retriever로 감싸기
book_retriever = pinecone_book.as_retriever(search_kwargs={"k": 8})

compression_retriever_book = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=book_retriever
)

# 도서 추천 검색 tool 정의
@tool
def search_book(query: str) -> List[Document]:
    """
    Search and access book recommendations tailored to specific majors or academic tracks,  
    along with brief summaries for each recommended title.

    Use this tool only for questions about books related to a student’s interests or field of study.
    """

    docs = compression_retriever_book.invoke(query)
    if len(docs) > 0:
        return docs
    
    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]

# 세특 관련 정보 검색
pinecone_seteuk = PineconeVectorStore.from_documents(
    documents=[], # 빈 리스트로 초기화
    index_name="college-admission-chatbot",   # 인덱스 이름
    embedding=embeddings,               # 임베딩 인스턴스
    pinecone_api_key=pinecone_api_key,
    namespace="seteuk", 
)

# 2. Pinecone 리트리버를 LangChain retriever로 감싸기
seteuk_retriever = pinecone_seteuk.as_retriever(search_kwargs={"k": 6})

# 3. 리랭커를 포함한 리트리버 생성
compression_retriever_seteuk = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=seteuk_retriever
)

# 서비스 검색
@tool
def search_seteuk(query: str) -> List[str]:
    """
    This database contains topic-related information for recommending 세특 (Detailed Learning and Performance Descriptions) activities.  
    It includes exploration topics and key keywords categorized by academic field, major, and subject.  
    This tool is used to recommend 세특 activity topics or retrieve related information.

    This tool is suitable for questions like:
    - I'm curious about 세특 topics.
    - Can you recommend an activity for my 세특?
    - What kind of exploration topic should I choose?

    """

    docs = compression_retriever_seteuk.invoke(query)
    if len(docs) > 0:
        return docs
    
    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]