import re
import json
import requests
import openai
from dotenv import load_dotenv
import os

# API 키 정보 로드
load_dotenv()

# API 키 읽어오기
openai_api_key = os.environ.get('OPENAI_API_KEY')

def slangword_translate(text: str, slang_dict: dict) -> str:
    """
    주어진 텍스트에서 슬랭(줄임말)을 모두 '(슬랭/정식표현)' 형태로 변환하는 함수

    처리 과정:
      1) 슬랭 키를 길이 내림차순으로 정렬하여 긴 키부터 매칭
      2) 특수문자를 이스케이프 처리하여 안전한 정규식 패턴 생성
      3) 한 번의 re.sub로 모든 매칭된 슬랭 치환

    Args:
        text (str): 원본 텍스트
        slang_dict (dict): {슬랭: 정식표현} 매핑 사전
    Returns:
        str: '(슬랭/정식표현)' 형태로 변환된 문자열
    """
    # 2-1) 키들을 길이 내림차순으로 정렬 (긴 키가 먼저 매칭되도록)
    sorted_slangs = sorted(slang_dict.keys(), key=len, reverse=True)
    escaped_keys  = [re.escape(s) for s in sorted_slangs]
    combined_re = re.compile('(' + '|'.join(escaped_keys) + ')')

    # 2-2) 치환 콜백: 매칭된 키(found) → "(found/formal)" 형태로 리턴
    def _repl(m: re.Match) -> str:
        found = m.group(1)         # 줄임말
        formal = slang_dict[found] # 정식 표현
        return f"({found}/{formal})"

    # 2-3) 한 번만 re.sub() 수행
    return combined_re.sub(_repl, text)


def select_contextual_word(input_translate: str) -> str:
    """
    중간 문자열 내 여러 '(슬랭/정식)' 패턴 각각에 대해 GPT-4o-mini 모델을 호출,
    문맥에 맞는 표현을 선택하여 괄호와 슬래시를 제거한 최종 문장을 반환하는 함수

    Args:
        input_translate (str): slangword_translate 출력 문자열
    Returns:
        str: 괄호·슬래시 제거 후 완성된 문장
    """
    system_prompt = (
        "당신은 줄임말과 정식표현을 구분하는 전문가입니다.\n"
        "다음 문장에는 여러 개의 괄호 안에 슬래시(/)로 병기된 두 어구(phrase)가 있습니다.\n"
        "각 괄호마다 두 어구 중 문맥상 더 적절한 어구 하나를 선택하여, \n"
        "전체 문장을 완성한 후 반환하세요.\n"
        "반드시 괄호와 슬래시를 제거하고, 선택된 어구 각각으로만 교체해야 합니다.\n"
        "그 외의 원래 텍스트(어미, 조사, 띄어쓰기 등)는 절대 변경하지 마세요."
        "당신은 오로지, 문맥에 맞는 표현을 선택하는 역할만 수행합니다.\n"
    )
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": input_translate}
        ]
    )
    return response.choices[0].message.content.strip()

def strip_slang_markers(intermediate: str) -> str:
    """
    '(슬랭/정식)' 형태의 모든 패턴을 정식표현만 남기고 제거
    나머지 문자열(구두점 및 공백 포함)은 원본 그대로 보존

    Args:
        intermediate (str): '(슬랭/정식)'이 포함된 중간 문자열
    Returns:
        str: 정식표현만 남은 최종 문자열
    """
    pattern = re.compile(r"\(([^/]+)/([^\)]+)\)")
    result_parts = []
    last_end = 0

    for m in pattern.finditer(intermediate):
        result_parts.append(intermediate[last_end:m.start()])
        result_parts.append(m.group(2))
        last_end = m.end()

    result_parts.append(intermediate[last_end:])
    return "".join(result_parts)


def replace_slang_word(text: str, slang_dict: dict) -> dict:
    """
    텍스트에 슬랭이 포함된 경우 아래 단계로 변환:
      1) slangword_translate → '(슬랭/정식)' 형태 생성
      2) 첫 '(.../...)'의 정식표현에 쉼표가 있으면 strip_slang_markers 사용
      3) 쉼표 없으면 select_contextual_word로 GPT 호출
    슬랭이 없으면 원문을 그대로 반환

    Args:
        text (str): 원본 텍스트
        slang_dict (dict): 슬랭-정식 매핑 사전
    Returns:
        dict: {'question': 최종 처리된 문자열}
    """

    intermediate = slangword_translate(text, slang_dict)

    if intermediate == text:
        return {"question": text}  # 🎯 치환 없으면 GPT 호출하지 않음

    m = re.search(r"\(([^/]+)/([^\)]+)\)", intermediate)
    if m:
        formal_part = m.group(2).strip()
        if "," in formal_part:
            result_text = strip_slang_markers(intermediate)
            return {"question": result_text}
    final_text = select_contextual_word(intermediate)
    return {"question": final_text}

