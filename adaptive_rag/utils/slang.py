import re
import json
import requests
import openai
from dotenv import load_dotenv
import os
from adaptive_rag.utils.state import AdaptiveRagState

# API 키 정보 로드
load_dotenv()

# API 키 읽어오기
openai_api_key = os.environ.get('OPENAI_API_KEY')

# ─────────────────────────────────────────────────────────────────────────────
# 1) GitHub에서 슬랭 사전 JSON을 가져와 파싱
# ─────────────────────────────────────────────────────────────────────────────
# url = "https://raw.githubusercontent.com/bdajiny/slang-dictionary/refs/heads/main/slang_dict.json"
# response = requests.get(url)
# raw_text = response.text
# slang_dict = json.loads(raw_text) # JSON 문자열을 파싱하여 딕셔너리로 변환!


# ─────────────────────────────────────────────────────────────────────────────
# 2) slangword_translate(): 한 번의 re.sub()로 모든 키-치환
# ─────────────────────────────────────────────────────────────────────────────
def slangword_translate(text: str, slang_dict: dict) -> str:
    """
    문장 내 모든 슬랭(줄임말)을 "(슬랭/정식표현)" 형태로 치환하여 반환합니다.
    - 키가 겹치는 경우(예: "국숭세" vs "국숭세단"), 길이 순(긴 것부터) 정렬 후 regex 생성
    """
    # 2-1) 키들을 길이 내림차순으로 정렬 (긴 키가 먼저 매칭되도록)
    sorted_slangs = sorted(slang_dict.keys(), key=len, reverse=True)
    escaped_keys  = [re.escape(s) for s in sorted_slangs]
    combined_re   = re.compile("(" + "|".join(escaped_keys) + ")")

    # 2-2) 치환 콜백: 매칭된 키(found) → "(found/formal)" 형태로 리턴
    def _repl(m: re.Match) -> str:
        found = m.group(1)         # 예: "과탐" 또는 "사탐"
        formal = slang_dict[found] # 예: "과학탐구영역" 또는 "사회탐구영역"
        return f"({found}/{formal})"

    # 2-3) 한 번만 re.sub() 수행
    return combined_re.sub(_repl, text)


# ─────────────────────────────────────────────────────────────────────────────
# 3) select_contextual_word(): 
#    문장 내 여러 개의 "(슬랭/정식)" 각각에 대해 둘 중 하나를 선택하여
#    전체 문장을 완성하는 프롬프트
# ─────────────────────────────────────────────────────────────────────────────
def select_contextual_word(input_translate: str) -> str:
    """
    GPT-4o-mini 모델을 사용하여, input_translate 문자열 안에 여러 개의
    '(슬랭/정식)' 형태가 있을 때, 각 괄호마다 문맥에 맞는 어구 하나씩을
    선택하여 전체 문장을 완성하고 반환합니다.
    - 괄호가 여러 개일 수 있으며, 각각에 대해 둘 중 하나를 선택해야 합니다.
    - 출력 시 괄호와 슬래시(/)는 모두 제거하고, 나머지 원문 텍스트를 보존합니다.
    - 원래 문장의 어미/조사 등은 변경하지 마세요.
    """
    system_prompt = (
        "다음 문장에는 여러 개의 괄호 안에 슬래시(/)로 병기된 두 어구(phrase)가 있습니다.\n"
        "각 괄호마다 두 어구 중 문맥상 더 적절한 어구 하나를 선택하여, \n"
        "전체 문장을 완성한 후 반환하세요.\n"
        "반드시 괄호와 슬래시를 제거하고, 선택된 어구 각각으로만 교체해야 합니다.\n"
        "그 외의 원래 텍스트(어미, 조사, 띄어쓰기 등)는 절대 변경하지 마세요."
    )
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": input_translate}
        ]
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4) replace_slang_word(): 슬랭 포함 시 (1)→(2)→(3) 순으로 처리
# ─────────────────────────────────────────────────────────────────────────────
def replace_slang_word(text: str, slang_dict: dict) -> dict:
    """
    1) text에 슬랭이 하나라도 있으면:
       1-1) slangword_translate(text) → "(슬랭/정식)" 형태의 중간 문자열(intermediate) 생성
       1-2) intermediate에서 "(.../...)" 내부의 '정식 부분'에 쉼표(,)가 있으면:
            - 그 "(슬랭/정식)" 패턴만 '정식'으로 대체하고, 나머지 텍스트 그대로 보존
       1-3) 쉼표가 없는 경우:
            - select_contextual_word(intermediate)로 GPT에게 “여러 괄호 각각에 대해 둘 중 하나씩”
              선택한 전체 문장을 반환하도록 요청.
            - 반환된 문장을 그대로 리턴
    2) text에 슬랭이 없으면 {"question": text} 반환
    """
    # (1) text에 슬랭이 포함되어 있는지 확인
    if any(s in text for s in slang_dict):
        # (1-1) "(슬랭/정식)" 형태로 치환 → 중간 문자열
        intermediate = slangword_translate(text, slang_dict)
        # ex) "저는 (과탐/과학탐구영역)과 (사탐/사회탐구영역)을 모두 신청했습니다."

        # (1-2) 각 "(.../...)"에서 slash 뒤 정식(formal_part) 부분 추출
        #     → 첫 번째 매칭만 검사 (정규표현식), 쉼표 포함 여부 확인
        m = re.search(r"\(([^/]+)/([^\)]+)\)", intermediate)
        if m:
            formal_part = m.group(2).strip()
            if "," in formal_part:
                # 쉼표가 포함된 경우:
                # "(슬랭/정식)" 전부를 "정식"으로 바꾸고 나머지 텍스트 보존
                result_text = re.sub(r"\([^/]+/([^\)]+)\)", r"\1", intermediate)
                return {"question": result_text}

        # (1-3) 쉼표가 없다면: GPT에게 “여러 괄호 각각에 대해” 선택을 요청
        final_text = select_contextual_word(intermediate)
        return {"question": final_text}

    # (2) 슬랭이 없으면: 원문 그대로 반환
    else:
        return {"question": text}