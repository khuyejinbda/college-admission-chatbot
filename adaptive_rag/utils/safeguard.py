from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
from adaptive_rag.utils.state import AdaptiveRagState

# 최초 1회만 실행해서 모델 로딩
def load_unsmile_pipeline(device: int = -1):
    model_name = 'smilegate-ai/kor_unsmile'
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,  # -1 for CPU, 0 for GPU
        return_all_scores=True,
        function_to_apply='sigmoid'
    )

unsmile_pipe = load_unsmile_pipeline(device=-1)

def profanity_prevention(
    state: dict,
    pipe: TextClassificationPipeline = unsmile_pipe
) -> dict:
    question = state.get("question", "")
    if not question.strip():
        return state

    result = pipe(question)[0]
    scores = {r['label']: r['score'] for r in result}

    if scores.get('악플/욕설', 0) > 0.5 or scores.get('clean', 1) < 0.3:
        new_state = {
            **state,
            "generation": """죄송합니다. 입력하신 질문에 부적절한 표현이 포함되어 있어 답변을 드릴 수 없습니다. 😰
욕설이나 비속어 없이 다시 질문해 주시기 바랍니다. 🙏🏻""",
            "stop": True
        }

        return new_state

    return state


def check_profanity_result(state):
    if state.get("stop"):
        return "__end__"
    return "route_question_adaptive"