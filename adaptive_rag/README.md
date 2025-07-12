# adaptive_rag
: 챗봇의 구조 설계 및 실행을 담당하는 핵심 모듈입니다.  
LangGraph 기반의 Adaptive RAG(검색 기반 응답 생성) 파이프라인을 구성하며, 다양한 기능 유틸리티가 포함되어 있습니다.

## 주요 구성 파일 (utils 폴더)

| 파일명           | 설명 |
|------------------|------|
| `tools.py`       | 검색 기능을 위한 `search_tool` 정의 |
| `state.py`       | LangGraph 기반 챗봇의 상태(state) 정의 |
| `slang.py`       | 사용자 입력의 줄임말을 처리하는 로직 |
| `safeguard.py`   | 욕설 및 부적절한 표현 필터링 |
| `mongoDB.py`     | 대화 로그를 MongoDB에 저장 |
| `router.py`      | 입력 질문을 처리 흐름에 따라 라우팅 |
| `search.py`      | `search_tool`을 활용한 문서 검색 수행 |
| `memory.py`      | 대화 이력을 LangChain 메모리에 저장 |
| `generate.py`    | 검색된 문서를 기반으로 답변 생성 |
| `pipeline.py`    | 전체 그래프를 컴파일하고 실행하는 파이프라인 정의 |

## ⚙️ 실행 방법

1.  **가상 환경 활성화 (권장)**:
    ```bash
    # 예시 (conda 사용 시)
    # conda activate your_env_name
    # 예시 (venv 사용 시)
    # source path/to/your_env/bin/activate
    ```

2.  **의존성 패키지 설치**:
    프로젝트 루트 디렉토리에 `requirements.txt` 파일이 있는지 확인하고, 다음 명령어를 실행하여 필요한 패키지를 설치.
    ```bash
    pip install -r requirements.txt
    ```

3. **필요한 환경 변수 (API KEY)**:
    아래 API 키들을 환경 변수로 등록하거나 .env 파일에 저장.
    - PINECONE_API_KEY
    - OPENAI_API_KEY
    - COHERE_API_KEY
    - MONGODB_API_KEY

## 참고
- 이 디렉터리는 챗봇 전체 파이프라인의 핵심 로직을 담고 있으며, 문서 검색 → 문맥 생성 → 답변 생성 흐름을 포함합니다.
- 테스트는 adaptive_rag.ipynb를 참고해 실행할 수 있습니다.
