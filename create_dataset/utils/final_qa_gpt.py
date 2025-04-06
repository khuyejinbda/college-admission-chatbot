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


final_system = """
Original QA pair:
QUESTION: {original_question}
ANSWER: {original_answer}

Critique of this QA pair:
{critique}

Your task is to improve the original QA pair based on the critique provided.

You are a Teacher/Professor in {domain}.
Your task is to refine the original question and answer based on the critique provided.
The purpose of the improved question and answer is to better test the understanding of the students.

QUESTION and ANSWER should be written in Korean. Response in JSON format which contains the question and answer.
DO NOT USE List in JSON format.
ANSWER should be a complete sentence.

#Format:
```json
{{
    "QUESTION": "개선된 질문",
    "ANSWER": "개선된 답변"
}}
```
"""

def custom_json_parser(text: str) -> Dict[str, str]:
    """
    GPT API의 응답에서 JSON 형식의 최종 QA를 추출하는 함수
    
    Parameters:
        text (str): GPT API의 응답 텍스트
    
    Returns:
        Dict[str, str]: 최종 QA 딕셔너리
    """
    # JSON 형식 추출을 위한 정규식 패턴
    pattern = r'\{\s*"QUESTION":\s*"([^"]+)",\s*"ANSWER":\s*"([^"]+)"\s*\}'
    
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
                        "ANSWER": match.group(2)
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
                    "ANSWER": match.group(2)
                }
            
        # 3. 모든 방법이 실패한 경우 원본 텍스트 반환
        return {
            "QUESTION": "파싱 오류",
            "ANSWER": text
        }
    
    except Exception as e:
        print(f"JSON 파싱 오류: {e}")
        # JSON 파싱에 실패한 경우 텍스트를 그대로 반환
        return {
            "QUESTION": "파싱 오류",
            "ANSWER": text
        }

def to_GPT_final(critique_qa: Dict[str, str], domain: str, model_type: str = "gpt-4o") -> Dict[str, str]:
    """
    OpenAI GPT API를 호출하여 최종 QA를 생성하는 함수
    
    Parameters:
        critique_qa (Dict[str, str]): 원본 QA와 비판을 담은 딕셔너리
        domain (str): 도메인 (예: 고교학점제)
        model_type (str): 사용할 GPT 모델
    
    Returns:
        Dict[str, str]: 최종 QA 딕셔너리
    """
    
    # OpenAI 클라이언트 생성
    client = OpenAI()
    
    # 포맷된 시스템 프롬프트 생성
    formatted_system = final_system.format(
        original_question=critique_qa["QUESTION"],
        original_answer=critique_qa["ANSWER"],
        critique=critique_qa.get("CRITIQUE", "No critique provided."),
        domain=domain
    )
    
    # GPT API 호출
    response = client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": formatted_system},
            {"role": "user", "content": "Refine the original QA pair based on the critique."}
        ],
        max_tokens=1000,
        temperature=0
    )
    
    # 응답에서 텍스트 추출
    response_text = response.choices[0].message.content
    
    # JSON 파싱
    final_qa = custom_json_parser(response_text)
    
    return final_qa

# 메인 로직
def generate_final_qa_pairs(critique_qa_pairs: List[Dict[str, str]], domain: str) -> List[Dict[str, str]]:
    """
    비판이 포함된 QA 쌍에서 최종 QA를 생성하는 함수
    
    Parameters:
        critique_qa_pairs (List[Dict[str, str]]): 비판이 포함된 QA 쌍 목록
        domain (str): 도메인 (예: 고교학점제)
    
    Returns:
        List[Dict[str, str]]: 최종 QA 쌍 목록
    """
    # 최종 QA 쌍을 저장할 리스트 초기화
    final_qa_pairs = []
    
    # 각 비판이 포함된 QA 쌍에 대해 최종 QA 생성
    for critique_qa in critique_qa_pairs:
        if critique_qa:
            # GPT API 호출하여 최종 QA 생성
            final_qa = to_GPT_final(critique_qa, domain)
            
            # 결과 추가
            final_qa_pairs.append(final_qa)
    
    return final_qa_pairs