"""
데이터 로딩 및 전처리 모듈
========================
역할: HuggingFace에서 데이터셋을 받아와서 모델이 학습할 수 있는 형태로 가공

전체 흐름:
  1. load_and_split()  — HF에서 데이터 다운로드 → train/val 분할
  2. apply_chat_template() — 각 행을 Qwen chat 형식 텍스트로 변환
  3. prepare_dataset()  — 위 1~2를 한번에 실행하는 파이프라인

핵심 개념:
  - SFT (Supervised Fine-Tuning)에서는 "프롬프트 + 정답 응답"을 하나의 텍스트로 만들어서 학습
  - 모델은 이 텍스트의 다음 토큰을 예측하는 방식으로 학습됨
  - SFTTrainer는 assistant 응답 부분만 loss를 계산함 (프롬프트 부분은 무시)
"""

import json
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


# ============================================================
# 프롬프트 템플릿
# ============================================================
# 학습 시와 추론 시 동일한 프롬프트를 사용해야 일관된 결과를 얻음
# → 이 상수들을 data_loader.py에 정의하고 다른 스크립트에서도 import해서 사용

SYSTEM_PROMPT = "당신은 한국어 감성 분석 전문가입니다. 주어진 텍스트의 감성을 분석하고 결과를 JSON으로 반환하세요."

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
    # probability가 문자열("0.95")로 저장되어 있으므로 float으로 변환
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


def load_and_split(dataset_name: str, train_ratio: float = 0.8, seed: int = 42) -> DatasetDict:
    """데이터셋 로드 후 train/validation 분할

    - HuggingFace Hub에서 데이터를 자동으로 다운로드하고 캐싱
    - 첫 실행 시만 다운로드, 이후는 로컬 캐시(~/.cache/huggingface/) 사용
    - seed를 고정해서 매번 같은 분할을 보장 (실험 재현성)
    """
    ds = load_dataset(dataset_name, split="train")
    split = ds.train_test_split(test_size=1 - train_ratio, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


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

    SFTTrainer는 이 "text" 컬럼을 읽어서 학습 데이터로 사용함.
    """

    def _format(example):
        messages = format_chat_messages(example)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    return dataset_dict.map(_format)


def prepare_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> DatasetDict:
    """전체 데이터 준비 파이프라인

    train.py에서 이 함수 하나만 호출하면 학습 데이터 준비 완료.
    반환값: {"train": Dataset(120건), "validation": Dataset(30건)}
    각 Dataset에는 "text" 컬럼이 추가되어 있음.
    """
    dataset_dict = load_and_split(dataset_name, train_ratio, seed)
    dataset_dict = apply_chat_template(dataset_dict, tokenizer)
    return dataset_dict
