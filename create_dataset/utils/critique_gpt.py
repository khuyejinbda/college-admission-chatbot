from dotenv import load_dotenv
import os
import openai

# 환경변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.environ.get("OPENAI_API_KEY")

# OpenAI 클라이언트 생성
client = openai.OpenAI()

critique_system = """
    Context information: The following QA pair is generated based on a given context.
    
    QUESTION: {QUESTION}
    ANSWER: {ANSWER}

    You are a Teacher/Professor in {domain}. Evaluate the QA pair according to the following criteria:
    1. Does the question accurately reflect the domain's key concepts?
    2. Is the answer complete and factually correct based solely on the provided context?
    3. Are the sentences clear and grammatically correct?

    Provide a concise critique with suggestions for improvement if necessary.

    CRITIQUE should be written in Korean. response in JSON format which contains the `question` and `answer` and 'critique'.
    CRITIQUE should be a complete sentence.
    
    Return your response in pure JSON format without any additional text, following this exact structure.
    
    #Format:
    ```json
    {{
        "QUESTION": "고교학점제의 장점은 무엇인가요?",
        "ANSWER": "학생이 주어진 과목을 학습하는 것이 아니라, 자신의 진로와 적성에 맞는 과목을 직접 선택해 학습함으로써 공부에 대한 흥미와 학습 동기를 높일 수 있다는 것이 가장 큰 장점입니다",
        "CRITIQUE": "질문과 답변이 고교학점제의 핵심 개념을 잘 반영하고 있습니다. 답변은 고교학점제의 주요 장점을 명확하게 설명하고 있으나, 추가적인 장점들(예: 자기주도적 학습 능력 향상, 진로 탐색 기회 확대 등)을 포함하면 더 완성도 높은 답변이 될 것입니다."
    }},
    {{
        "QUESTION": "그동안 운영되어 온 고교학점제와 올해부터 전면 시행되는 고교학점제의 차이점은 무엇인가요?",
        "ANSWER": "그간 고교학점제 전면 시행에 대비하여 고교 현장에서 학생 선택형 교육과정 운영 등 학점제 요소를 일부 적용해 왔습니다. 다만, 지금까지 학년별 수업일 수 기준(수업일 수의 2/3이상 출석)만 충족하면 고등학교 졸업이 가능해 학점 취득과 졸업 자격 획득이 연계되지 않았습니다. 하지만 올해 신입생부터는 3년간 192학점 이상의 학점도 취득해야 졸업이 가능해집니다",
        "CRITIQUE": "질문과 답변 모두 정확하고 명확합니다. 답변은 기존 고교학점제와 전면 시행되는 고교학점제의 가장 큰 차이점인 졸업 요건의 변화를 잘 설명하고 있습니다. 다만, 문장 끝에 마침표가 누락되어 있으니 추가하면 좋겠습니다."
    }},
    {{
        "QUESTION": "고교-대학 연계 학점인정 과목이 무엇인가요?",
        "ANSWER": "시도교육청과 협약한 지역대학이 개설하여 운영하는 과목으로서 학교 밖 교육의 한 유형에 해당하는 과목입니다. 해당 과목을 이수한 학생은 고등학교의 학점뿐만 아니라 추후 학생이 해당 대학에 진학할 경우 대학의 학점으로도 인정받을 수 있습니다.",
        "CRITIQUE": "질문과 답변 모두 고교-대학 연계 학점인정 과목의 개념을 정확히 설명하고 있습니다. 답변은 이 과목의 정의와 학생들이 얻을 수 있는 혜택을 명확하게 제시하고 있어 적절합니다. 문법적으로도 오류가 없고 이해하기 쉽게 작성되었습니다."
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

# 비판 시스템 프롬프트
critique_system = """
Context information: The following QA pair is generated based on a given context.

QUESTION: {QUESTION}
ANSWER: {ANSWER}

You are a Teacher/Professor in {domain}. Evaluate the QA pair according to the following criteria:
1. Does the question accurately reflect the domain's key concepts?
2. Is the answer complete and factually correct based solely on the provided context?
3. Are the sentences clear and grammatically correct?

Provide a concise critique with suggestions for improvement if necessary.

CRITIQUE should be written in Korean. response in JSON format which contains the `question` and `answer` and 'critique'.
CRITIQUE should be a complete sentence.

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

