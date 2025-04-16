import anthropic
import os
import json
import re
from dotenv import load_dotenv
from typing import List, Dict, Any

# 환경변수 로드
load_dotenv()

# Anthropic API 키 설정
api_key = os.environ.get("ANTHROPIC_API_KEY")

QA_system = """Context information is below. You are only aware of this context and nothing else.
---------------------

{context}

---------------------
Based on the following context that has been converted into Markdown, generate only questions based on the below query.

You are a Teacher/Professor in the field of {domain}.
Your role is to generate realistic and natural questions that students or parents—who may not be familiar with the admissions process—are likely to ask, and to provide clear, accurate answers to those questions.

Your task is to provide exactly **{num_questions}** question(s) for an upcoming quiz/examination. 
You are not to provide more or less than this number of questions. 


You will be given a piece of text extracted from the context. The context may include:
- Descriptions of high school subjects
- Guidelines for school operations (e.g., 고교학점제)
- Preferred subject lists for university admissions
- Actual Q&A content provided by education offices or schools
- **Table Information Containing College Admissions Data**

You must also provide the answer to each question. The answer should be based on the context information provided only.

If a table is provided as input, please analyze the **Markdown table structure** and extract its information accordingly. When extracting information, do not include text that cannot be understood without external information (e.g., '택 1' etc) as is, but rather fully comprehend the information first and then transform it to make it easier to understand.

### The questions must meet the following criteria:
- They should sound like real questions students would ask (e.g., “이과 가려면 어떤 과목 들어야 해요?”)
- They must be based **only on the content provided** in the document (Do not guess, infer, or fabricate any information)
- They should focus on topics students truly care about, such as subject selection, graduation requirements, admissions, or career paths

### The answers must meet the following criteria:
- They must be written in full, friendly, and natural Korean sentences
- The content must be strictly grounded in the given document (no assumptions or extra info)
- They should provide helpful, reassuring, and actionable guidance for the student


QUESTION and ANSWER should be written in Korean. response in JSON format which contains the `question` and `answer`.
DO NOT USE List in JSON format.
ANSWER should be a complete sentence.

#Format:
```json
{{
    "QUESTION": "고교학점제의 장점은 무엇인가요?",
    "ANSWER": "학생이 주어진 과목을 학습하는 것이 아니라, 자신의 진로와 적성에 맞는 과목을 직접 선택해 학습함으로써 공부에 대한 흥미와 학습 동기를 높일 수 있다는 것이 가장 큰 장점입니다"
}},
{{
    "QUESTION": "그동안 운영되어 온 고교학점제와 올해부터 전면 시행되는 고교학점제의 차이점은 무엇인가요?",
    "ANSWER": "그간 고교학점제 전면 시행에 대비하여 고교 현장에서 학생 선택형 교육과정 운영 등 학점제 요소를 일부 적용해 왔습니다. 다만, 지금까지 학년별 수업일 수 기준(수업일 수의 2/3이상 출석)만 충족하면 고등학교 졸업이 가능해 학점 취득과 졸업 자격 획득이 연계되지 않았습니다. 하지만 올해 신입생부터는 3년간 192학점 이상의 학점도 취득해야 졸업이 가능해집니다"    
}},
{{
    "QUESTION": "고교-대학 연계 학점인정 과목이 무엇인가요?",
    "ANSWER": "시도교육청과 협약한 지역대학이 개설하여 운영하는 과목으로서 학교 밖 교육의 한 유형에 해당하는 과목입니다. 해당 과목을 이수한 학생은 고등학교의 학점뿐만 아니라 추후 학생이 해당 대학에 진학할 경우 대학의 학점으로도 인정받을 수 있습니다."    
}}
```
"""

def custom_json_parser(text: str) -> List[Dict[str, str]]:
    """
    Claude API의 응답에서 JSON 형식의 QA 쌍을 추출하는 함수
    
    Parameters:
        text (str): Claude API의 응답 텍스트
    
    Returns:
        List[Dict[str, str]]: QA 쌍 목록
    """
    # JSON 형식 추출을 위한 정규식 패턴
    pattern = r'\{\s*"QUESTION":\s*"([^"]+)",\s*"ANSWER":\s*"([^"]+)"\s*\}'
    
    try:
        # 1. 먼저 텍스트에서 ```json과 ``` 사이의 내용을 추출
        json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        
        if json_block_match:
            json_text = json_block_match.group(1)
            
            # 2. 중괄호를 기준으로 분할
            matches = re.finditer(pattern, json_text)
            
            result = []
            for match in matches:
                question = match.group(1)
                answer = match.group(2)
                result.append({"QUESTION": question, "ANSWER": answer})
            
            return result
        
        # 코드 블록이 없는 경우, 텍스트 전체에서 패턴 검색
        matches = re.finditer(pattern, text)
        
        result = []
        for match in matches:
            question = match.group(1)
            answer = match.group(2)
            result.append({"QUESTION": question, "ANSWER": answer})
        
        return result
    
    except Exception as e:
        print(f"JSON 파싱 오류: {e}")
        # JSON 파싱에 실패한 경우 텍스트를 그대로 반환
        return [{"QUESTION": "파싱 오류", "ANSWER": text}]

def to_Claude(prompt: Dict[str, str], model_type: str = "claude-3-7-sonnet-20250219") -> List[Dict[str, str]]:
    """
    Anthropic Claude API를 호출하고 결과를 LangChain과 유사한 형식으로 반환하는 함수
    
    Parameters:
        prompt (Dict[str, str]): 컨텍스트, 도메인, 질문 수를 담은 딕셔너리
        model_type (str): 사용할 Claude 모델
    
    Returns:
        List[Dict[str, str]]: QA 쌍 목록
    """
    
    # Anthropic 클라이언트 생성
    client = anthropic.Anthropic()
    
    # 포맷된 시스템 프롬프트 생성
    formatted_system = QA_system.format(
        context=prompt["context"],
        domain=prompt["domain"],
        num_questions=prompt["num_questions"]
    )
    
    # Claude API 호출
    response = client.messages.create(
        model=model_type,
        system=formatted_system,
        messages=[
            {"role": "user", "content": "Generate questions and answers based on the context."}
        ],
        max_tokens=3000,
        temperature=0
    )
    
    # 응답에서 텍스트 추출
    response_text = response.content[0].text
    
    # JSON 파싱
    qa_pairs = custom_json_parser(response_text)
    
    return qa_pairs

# 메인 로직
def generate_qa_pairs(elements: List[str], domain: str, num_questions: str) -> List[Dict[str, str]]:
    """
    여러 텍스트 청크에서 QA 쌍을 생성하는 함수
    
    Parameters:
        elements (List[str]): 텍스트 청크 목록
        domain (str): 도메인 (예: 고교학점제)
        num_questions (str): 각 청크당 생성할 질문 수
    
    Returns:
        List[Dict[str, str]]: 생성된 모든 QA 쌍 목록
    """
    # 질문-답변 쌍을 저장할 리스트 초기화
    qa_pairs = []
    
    # 각 텍스트 청크에 대해 질문-답변 쌍 생성
    for element in elements:
        if element:
            # Claude API 호출하여 QA 쌍 생성
            qa_batch = to_Claude({
                "context": element,  # 텍스트 청크
                "domain": domain,    # 질문 생성 도메인
                "num_questions": num_questions  # 청크당 생성할 질문 수
            })
            
            # 결과 추가
            qa_pairs.extend(qa_batch)
    
    return qa_pairs