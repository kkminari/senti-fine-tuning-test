"""
데이터 로딩 및 전처리 모듈
========================
역할: HuggingFace에서 데이터셋을 받아와서 모델이 학습할 수 있는 형태로 가공

전체 흐름:
  1. load_and_split()  — HF에서 데이터 다운로드 → train/val/test 분할
  2. apply_chat_template() — 각 행을 Qwen chat 형식 텍스트로 변환
  3. prepare_dataset()  — 위 1~2를 한번에 실행하는 파이프라인

핵심 개념:
  - SFT (Supervised Fine-Tuning)에서는 "프롬프트 + 정답 응답"을 하나의 텍스트로 만들어서 학습
  - 모델은 이 텍스트의 다음 토큰을 예측하는 방식으로 학습됨
  - SFTTrainer는 assistant 응답 부분만 loss를 계산함 (프롬프트 부분은 무시)
"""

import re
import json
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


# ============================================================
# 프롬프트 템플릿
# ============================================================
# [1-1 개선] JSON 스키마와 규칙을 명시하여 모델이 출력 형식을 정확히 학습하도록 함
# - 기존: "JSON으로 반환하세요" (모호함)
# - 개선: 정확한 키 이름, 값 범위, 규칙을 프롬프트에 포함
# - 효과: 적은 데이터로도 출력 형식 학습이 빨라지고, 추론 시 일관성 향상

SYSTEM_PROMPT = """당신은 한국어 감성 분석 전문가입니다. 주어진 텍스트의 감성을 분석하고 아래 JSON 형식으로만 반환하세요.

출력 형식:
{"sentiment": "positive|negative|neutral", "probability": 0.0~1.0, "positive_topics": [...], "negative_topics": [...]}

규칙:
- sentiment는 반드시 positive, negative, neutral 중 하나
- probability는 해당 감성의 확신도 (0.0~1.0 사이 소수)
- positive_topics: 텍스트에서 긍정적으로 언급된 핵심 키워드 리스트
- negative_topics: 텍스트에서 부정적으로 언급된 핵심 키워드 리스트
- 해당 토픽이 없으면 빈 리스트 []
- JSON만 출력하고 다른 텍스트는 포함하지 마세요"""

USER_TEMPLATE = '다음 텍스트의 감성을 분석하세요:\n"{text}"'


def parse_topics(topic_str: str) -> list[str]:
    """쉼표 구분 토픽 문자열을 리스트로 변환

    원본 데이터 예시: "색상, 향, 발림" → ["색상", "향", "발림"]
    빈 문자열이나 None → []
    """
    if not topic_str or topic_str.strip() == "":
        return []
    return [t.strip() for t in topic_str.split(",") if t.strip()]


def build_output_json(example: dict) -> str:
    """데이터 행을 JSON 출력 문자열로 변환

    모델이 생성해야 할 "정답" 응답을 만드는 함수.
    학습 데이터의 각 행(Input, sentiment, probability, topics)을
    아래와 같은 JSON 문자열로 변환:

    {"sentiment": "positive", "probability": 0.95,
     "positive_topics": ["색상", "향"], "negative_topics": []}

    ensure_ascii=False: 한국어가 \\uXXXX로 변환되지 않도록 방지
    """
    prob = float(example["probability"]) if isinstance(example["probability"], str) else example["probability"]
    return json.dumps(
        {
            "sentiment": example["sentiment"],
            "probability": prob,
            "positive_topics": parse_topics(example.get("positive_topics", "")),
            "negative_topics": parse_topics(example.get("negative_topics", "")),
        },
        ensure_ascii=False,
    )


def format_chat_messages(example: dict) -> list[dict]:
    """Qwen chat 형식의 메시지 리스트 생성

    Qwen 모델은 OpenAI 스타일의 chat 메시지를 사용:
      - system: 모델의 역할/지시사항 설정
      - user: 사용자 입력 (분석할 텍스트)
      - assistant: 모델이 생성해야 할 정답 (학습용)

    학습 시에는 3개 메시지 모두 포함 (assistant = 정답)
    추론 시에는 system + user만 주고 assistant는 모델이 생성
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=example["Input"])},
        {"role": "assistant", "content": build_output_json(example)},
    ]


def load_and_split(
    dataset_name: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetDict:
    """데이터셋 로드 후 train/validation/test 3분할

    [1-2 개선] stratify_by_column으로 감성 라벨 비율 유지
    - 150건에서 랜덤 분할하면 neutral 같은 소수 클래스가 특정 split에서 0건이 될 수 있음
    - stratify를 적용하면 각 split에서 긍정/부정/중립 비율이 원본과 동일하게 유지됨

    [3-1 개선] test set 별도 분리
    - 기존: train/val 2분할 → val로 하이퍼파라미터 조정 + 최종 평가 동시 수행
    - 개선: train/val/test 3분할
      - val: 학습 중 과적합 모니터링 + 하이퍼파라미터 선택에 사용
      - test: 최종 성능 측정에만 사용 (1회만 측정)
    - 이유: val로 파라미터를 여러 번 조정하면 val에도 간접적으로 과적합됨
    """
    ds = load_dataset(dataset_name, split="train")

    # 1단계: train+val / test 분리
    split1 = ds.train_test_split(
        test_size=test_ratio,
        seed=seed,
        stratify_by_column="sentiment",  # [1-2] 감성 라벨 비율 유지
    )

    # 2단계: train / val 분리
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)  # 남은 데이터 중 val 비율
    split2 = split1["train"].train_test_split(
        test_size=val_ratio_adjusted,
        seed=seed,
        stratify_by_column="sentiment",
    )

    return DatasetDict({
        "train": split2["train"],
        "validation": split2["test"],
        "test": split1["test"],
    })


def strip_thinking(text: str) -> str:
    """[1-4] Qwen3 thinking mode 출력에서 <think>...</think> 블록 제거

    Qwen3는 기본적으로 답변 전에 사고 과정을 <think> 태그로 출력함.
    이 태그 안에 {, } 등이 포함될 수 있어 JSON 파싱을 방해하므로 제거.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def apply_chat_template(dataset_dict: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    """채팅 템플릿을 적용하여 "text" 컬럼 생성

    tokenizer.apply_chat_template()이 하는 일:
      메시지 리스트 → Qwen 전용 특수 토큰이 포함된 하나의 문자열

    예시 출력:
      <|im_start|>system
      당신은 한국어 감성 분석 전문가입니다...<|im_end|>
      <|im_start|>user
      다음 텍스트의 감성을 분석하세요:
      "색상 진짜 이뻐요..."<|im_end|>
      <|im_start|>assistant
      {"sentiment": "positive", ...}<|im_end|>

    add_generation_prompt=False:
      학습 데이터이므로 assistant 응답까지 포함.
      True로 하면 assistant 시작 토큰만 붙임 (추론용)

    [1-4] enable_thinking=False:
      Qwen3의 thinking mode를 비활성화하여 <think> 태그가 학습 데이터에 포함되지 않도록 함.
      학습 시 thinking이 활성화되면, 추론 시에도 thinking을 출력하려는 습관이 남음.

    SFTTrainer는 이 "text" 컬럼을 읽어서 학습 데이터로 사용함.
    """

    def _format(example):
        messages = format_chat_messages(example)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,  # [1-4] Qwen3 thinking 비활성화
        )
        return {"text": text}

    return dataset_dict.map(_format)


def prepare_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetDict:
    """전체 데이터 준비 파이프라인

    train.py에서 이 함수 하나만 호출하면 학습 데이터 준비 완료.
    반환값: {"train": Dataset, "validation": Dataset, "test": Dataset}
    각 Dataset에는 "text" 컬럼이 추가되어 있음.
    """
    dataset_dict = load_and_split(dataset_name, train_ratio, val_ratio, test_ratio, seed)
    dataset_dict = apply_chat_template(dataset_dict, tokenizer)
    return dataset_dict
