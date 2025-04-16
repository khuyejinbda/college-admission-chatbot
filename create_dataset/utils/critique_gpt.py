critique_system = """
Context information: The following QA pair is generated based on a given context.

QUESTION: {QUESTION}
ANSWER: {ANSWER}

You are a Teacher/Professor in {domain}. 

Evaluate the QA pair based on the criteria below and write a concise critique in Korean, using 1–2 full sentences.**

All questions and answers should be criticized.

---

Evaluation Criteria (in order of importance)
1. Is the answer appropriate and relevant to the question?
2. Are specific expressions (e.g., '선택권 B', '택 1', etc.) sufficiently understandable without external information?
3. Is the question too vague, or is the answer overly simplistic or repetitive?
4. Is the tone and structure appropriate and easy for students to understand?
5. Are questions and answers grammatical errors free and complete sentences? 

---

Guidelines for Writing the Critique
- Avoid repeating generic comments (e.g., “Well written”, “Clear”)
- Do not include vague phrases like “The question is good”
- Focus on main issues per critique and keep the explanation brief and specific
response in JSON format which contains the `question` and `answer` and `critique`.
CRITIQUE should be a complete sentence.

---

Return your response in pure JSON format without any additional text, following this exact structure.

#Format:
```json
{{
    "QUESTION": "고교학점제의 장점은 무엇인가요?",
    "ANSWER": "학생이 주어진 과목을 학습하는 것이 아니라, 자신의 진로와 적성에 맞는 과목을 직접 선택해 학습함으로써 공부에 대한 흥미와 학습 동기를 높일 수 있다는 것이 가장 큰 장점입니다",
    "CRITIQUE": "질문과 답변이 고교학점제의 핵심 개념을 잘 반영하고 있습니다. 답변은 고교학점제의 주요 장점을 명확하게 설명하고 있으나, 추가적인 장점들(예: 자기주도적 학습 능력 향상, 진로 탐색 기회 확대 등)을 포함하면 더 완성도 높은 답변이 될 것입니다."
}}
```
"""


import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 로드
load_dotenv()

# OpenAI API 키 설정
api_key = os.environ.get("OPENAI_API_KEY")


def custom_json_parser(text: str) -> Dict[str, str]:
    """
    GPT API의 응답에서 JSON 형식의 QA 비판을 추출하는 함수
    
    Parameters:
        text (str): GPT API의 응답 텍스트
    
    Returns:
        Dict[str, str]: QA 비판 딕셔너리
    """
    # JSON 형식 추출을 위한 정규식 패턴
    pattern = r'\{\s*"QUESTION":\s*"([^"]+)",\s*"ANSWER":\s*"([^"]+)",\s*"CRITIQUE":\s*"([^"]+)"\s*\}'
    
    try:
        # 1. 먼저 텍스트에서 ```json과 ``` 사이의 내용을 추출
        json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        
        if json_block_match:
            json_text = json_block_match.group(1)
            # JSON 파싱 시도
            try:
                # JSON 문자열에서 따옴표 이스케이프 처리
                cleaned_json = json_text.replace('\\"', '"')
                result = json.loads(cleaned_json)
                return result
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 정규식 사용
                match = re.search(pattern, json_text)
                if match:
                    return {
                        "QUESTION": match.group(1),
                        "ANSWER": match.group(2),
                        "CRITIQUE": match.group(3)
                    }
        
        # 2. 코드 블록이 없는 경우 직접 JSON 파싱 시도
        try:
            result = json.loads(text)
            return result
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 정규식 사용
            match = re.search(pattern, text)
            if match:
                return {
                    "QUESTION": match.group(1),
                    "ANSWER": match.group(2),
                    "CRITIQUE": match.group(3)
                }
            
        # 3. 모든 방법이 실패한 경우 원본 텍스트 반환
        return {
            "QUESTION": "파싱 오류",
            "ANSWER": "파싱 오류",
            "CRITIQUE": text
        }
    
    except Exception as e:
        print(f"JSON 파싱 오류: {e}")
        # JSON 파싱에 실패한 경우 텍스트를 그대로 반환
        return {
            "QUESTION": "파싱 오류",
            "ANSWER": "파싱 오류",
            "CRITIQUE": text
        }

def to_GPT(qa_pair: Dict[str, str], domain: str, model_type: str = "gpt-4o") -> Dict[str, str]:
    """
    OpenAI GPT API를 호출하고 QA 비판 결과를 반환하는 함수
    
    Parameters:
        qa_pair (Dict[str, str]): 질문과 답변을 담은 딕셔너리
        domain (str): 도메인 (예: 고교학점제)
        model_type (str): 사용할 GPT 모델
    
    Returns:
        Dict[str, str]: QA 비판 딕셔너리
    """
    
    # OpenAI 클라이언트 생성
    client = OpenAI()
    
    # 포맷된 시스템 프롬프트 생성
    formatted_system = critique_system.format(
        QUESTION=qa_pair["QUESTION"],
        ANSWER=qa_pair["ANSWER"],
        domain=domain
    )
    
    # GPT API 호출
    response = client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": formatted_system},
            {"role": "user", "content": "Evaluate the given QA pair and provide a critique."}
        ],
        max_tokens=1000,
        temperature=0
    )
    
    # 응답에서 텍스트 추출
    response_text = response.choices[0].message.content
    
    # JSON 파싱
    critique_result = custom_json_parser(response_text)
    
    return critique_result

# 메인 로직
def critique_qa_pairs(qa_pairs: List[Dict[str, str]], domain: str) -> List[Dict[str, str]]:
    """
    여러 QA 쌍에 대한 비판을 생성하는 함수
    
    Parameters:
        qa_pairs (List[Dict[str, str]]): QA 쌍 목록
        domain (str): 도메인 (예: 고교학점제)
    
    Returns:
        List[Dict[str, str]]: 비판이 추가된 QA 쌍 목록
    """
    # 비판 결과를 저장할 리스트 초기화
    critique_results = []
    
    # 각 QA 쌍에 대해 비판 생성
    for qa_pair in qa_pairs:
        if qa_pair:
            # GPT API 호출하여 비판 생성
            critique_result = to_GPT(qa_pair, domain)
            
            # 결과 추가
            critique_results.append(critique_result)
    
    return critique_results