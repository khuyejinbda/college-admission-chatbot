# 챗봇 애플리케이션 실행 가이드

이 문서는 Bearable 챗봇 애플리케이션의 프론트엔드와 백엔드를 실행하는 방법을 안내합니다.

## 사전 준비

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

## 애플리케이션 실행 단계

프론트엔드와 백엔드는 별도의 터미널에서 각각 실행.

### 1. 백엔드 서버 실행 (FastAPI)

FastAPI로 구현된 백엔드 서버를 먼저 실행.

1.  새 터미널을 엽니다.
2.  프로젝트 루트 디렉토리로 이동.
    ```bash
    cd path/to/your/bearable_chatbot
    ```
3.  다음 명령어를 실행하여 Uvicorn으로 FastAPI 애플리케이션을 시작.
    ```bash
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload`: 코드 변경 시 서버가 자동으로 재시작 (개발 시 유용).
    *   `--host 0.0.0.0`: 모든 네트워크 인터페이스에서 접속을 허용.
    *   `--port 8000`: 서버가 8000번 포트에서 실행.

    서버가 정상적으로 시작되면 다음과 유사한 로그가 터미널에 출력:
    ```
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
    INFO:     Started reloader process [...] using [...]
    INFO:     Started server process [...]
    INFO:     Waiting for application startup.
    Application startup: Initializing RAG graph...
    Initializing RAG graph for API...
    RAG graph initialized for API.
    RAG graph initialized and ready.
    INFO:     Application startup complete.
    ```

### 2. 프론트엔드 애플리케이션 실행 (Streamlit)

백엔드 서버가 실행 중인 상태에서 Streamlit으로 구현된 프론트엔드 애플리케이션을 실행.

1.  새 터미널을 엽니다.
2.  프로젝트 루트 디렉토리로 이동합니다.
    ```bash
    cd path/to/your/bearable_chatbot
    ```
3.  다음 명령어를 실행하여 Streamlit 애플리케이션을 시작.
    ```bash
    streamlit run frontend/app.py
    ```

    애플리케이션이 정상적으로 시작되면 웹 브라우저가 자동으로 열리거나, 터미널에 접속 가능한 URL(일반적으로 `http://localhost:8501`)이 표시.

    ```
    You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
    Network URL: http://<your-local-ip>:8501
    ```

## 사용 방법

1.  백엔드 서버와 프론트엔드 애플리케이션이 모두 실행 중인지 확인.
2.  웹 브라우저에서 Streamlit 애플리케이션 URL(예: `http://localhost:8501`)에 접속.

## 종료 방법

1.  각 터미널에서 `Ctrl+C`를 눌러 Streamlit 애플리케이션과 FastAPI 백엔드 서버를 각각 종료.
