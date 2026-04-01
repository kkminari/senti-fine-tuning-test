"""
평가 스크립트: base 모델 vs fine-tuned 모델 비교
================================================
실행: cd src && python evaluate.py

하는 일:
  1. Validation 30건에 대해 base Qwen3-14B와 fine-tuned 모델 각각 추론
  2. 6개 지표로 성능 비교:
     - json_parse_rate: 모델 출력이 유효한 JSON인지 (파인튜닝 효과의 첫 번째 지표)
     - accuracy: 감성(긍/부/중) 분류 정확도
     - f1_macro: 클래스별 F1의 평균 (불균형 데이터에서 중요)
     - prob_mae: 확률값 예측의 평균절대오차 (낮을수록 좋음)
     - pos/neg_topic_f1: 토픽 추출 정확도

기대되는 파인튜닝 효과:
  - base 모델: JSON 형식을 지키지 않거나, 감성 분류가 부정확할 수 있음
  - fine-tuned: JSON 형식 준수율 ↑, 감성 정확도 ↑, 토픽 추출 ↑
"""

import json
import yaml
import torch
from dotenv import load_dotenv

load_dotenv()

from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score

from data_loader import load_and_split, format_chat_messages, SYSTEM_PROMPT, USER_TEMPLATE


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_response(model, tokenizer, text: str, config: dict) -> str:
    """단일 텍스트에 대해 모델 응답 생성

    추론 과정:
      1. system + user 메시지로 프롬프트 구성
      2. add_generation_prompt=True → assistant 시작 토큰까지만 포함
         (학습 때는 False로 정답까지 포함, 추론 때는 True로 모델이 생성)
      3. model.generate()로 다음 토큰을 반복 생성
      4. 생성된 토큰만 디코딩 (프롬프트 부분은 제외)

    torch.no_grad(): 추론 시에는 gradient 계산 불필요 → 메모리 절약
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config["inference"]["max_new_tokens"],
            temperature=config["inference"]["temperature"],
            do_sample=True,           # temperature 기반 샘플링 활성화
            pad_token_id=tokenizer.pad_token_id,  # 패딩 토큰 경고 방지
        )
    # outputs[0]에서 입력 프롬프트 길이만큼 잘라내면 생성된 부분만 남음
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def parse_response(response: str) -> dict | None:
    """모델 응답을 JSON으로 파싱 시도

    모델이 항상 깨끗한 JSON만 출력하지는 않음.
    "분석 결과입니다: {"sentiment": ...}" 처럼 앞뒤에 텍스트가 붙을 수 있어서
    첫 번째 '{' ~ 마지막 '}'를 찾아서 파싱 시도.
    실패하면 None 반환 → json_parse_rate 지표에 반영.
    """
    try:
        if "{" in response and "}" in response:
            start = response.index("{")
            end = response.rindex("}") + 1
            return json.loads(response[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def compute_topic_f1(pred_topics: list[str], gold_topics: list[str]) -> dict:
    """토픽 리스트 간 precision, recall, F1 계산

    예시: gold=["색상", "향", "발림"], pred=["색상", "발림", "가격"]
      - precision = 2/3 (예측한 것 중 정답 비율)
      - recall = 2/3 (정답 중 맞춘 비율)
      - F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 0.667

    Counter를 사용하는 이유: 중복 토픽이 있을 수 있어서 set보다 정확
    """
    if not gold_topics and not pred_topics:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gold_topics or not pred_topics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_set = Counter(pred_topics)
    gold_set = Counter(gold_topics)
    common = sum((pred_set & gold_set).values())  # 교집합의 원소 수

    precision = common / sum(pred_set.values()) if pred_set else 0.0
    recall = common / sum(gold_set.values()) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_model(model, tokenizer, dataset, config: dict, label: str = "model") -> dict:
    """validation 데이터 전체에 대해 모델 평가 실행"""
    gold_sentiments = []
    pred_sentiments = []
    prob_errors = []
    pos_topic_f1s = []
    neg_topic_f1s = []
    json_success = 0
    total = len(dataset)

    print(f"\n{'='*60}")
    print(f"평가 시작: {label} ({total}건)")
    print(f"{'='*60}")

    for i, example in enumerate(dataset):
        response = generate_response(model, tokenizer, example["Input"], config)
        parsed = parse_response(response)

        if parsed is not None:
            json_success += 1

            # 감성 분류 비교
            gold_sentiments.append(example["sentiment"])
            pred_sentiments.append(parsed.get("sentiment", "unknown"))

            # 확률값 MAE (Mean Absolute Error)
            # 예: 정답 0.95, 예측 0.87 → 오차 = |0.95 - 0.87| = 0.08
            gold_prob = float(example["probability"]) if isinstance(example["probability"], str) else example["probability"]
            pred_prob = parsed.get("probability", 0.0)
            if isinstance(pred_prob, (int, float)):
                prob_errors.append(abs(gold_prob - pred_prob))

            # 토픽별 F1
            gold_pos = [t.strip() for t in (example.get("positive_topics", "") or "").split(",") if t.strip()]
            gold_neg = [t.strip() for t in (example.get("negative_topics", "") or "").split(",") if t.strip()]
            pred_pos = parsed.get("positive_topics", [])
            pred_neg = parsed.get("negative_topics", [])
            if isinstance(pred_pos, list):
                pos_topic_f1s.append(compute_topic_f1(pred_pos, gold_pos)["f1"])
            if isinstance(pred_neg, list):
                neg_topic_f1s.append(compute_topic_f1(pred_neg, gold_neg)["f1"])
        else:
            # JSON 파싱 실패 → 감성 분류도 실패로 처리
            gold_sentiments.append(example["sentiment"])
            pred_sentiments.append("parse_fail")

        if (i + 1) % 10 == 0:
            print(f"  진행: {i+1}/{total}")

    # 지표 계산
    results = {
        "label": label,
        "total": total,
        "json_parse_rate": json_success / total if total > 0 else 0.0,
        "accuracy": accuracy_score(gold_sentiments, pred_sentiments),
        # f1_macro: 각 클래스의 F1을 구한 뒤 단순 평균
        #   → 데이터가 불균형해도 소수 클래스에 동일한 가중치
        "f1_macro": f1_score(gold_sentiments, pred_sentiments, average="macro", zero_division=0),
        # f1_weighted: 각 클래스의 샘플 수에 비례하여 가중 평균
        "f1_weighted": f1_score(gold_sentiments, pred_sentiments, average="weighted", zero_division=0),
        "prob_mae": sum(prob_errors) / len(prob_errors) if prob_errors else float("nan"),
        "pos_topic_f1": sum(pos_topic_f1s) / len(pos_topic_f1s) if pos_topic_f1s else float("nan"),
        "neg_topic_f1": sum(neg_topic_f1s) / len(neg_topic_f1s) if neg_topic_f1s else float("nan"),
    }

    print(f"\n--- 결과: {label} ---")
    for k, v in results.items():
        if k == "label":
            continue
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return results


def main():
    config = load_config()
    model_name = config["model"]["name"]
    adapter_dir = config["output"]["adapter_dir"]

    # validation 데이터만 사용 (학습에 쓰지 않은 30건)
    data_cfg = config["data"]
    dataset_dict = load_and_split(data_cfg["dataset_name"], data_cfg["train_ratio"], data_cfg["seed"])
    val_dataset = dataset_dict["validation"]

    # 양자화 설정 (학습 때와 동일하게)
    compute_dtype = getattr(torch, config["quantization"]["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Base 모델 평가 ===
    # 파인튜닝 전 원본 모델의 성능을 먼저 측정 (비교 기준선)
    print("\nBase 모델 로딩 중...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_results = evaluate_model(base_model, tokenizer, val_dataset, config, label="Base Qwen3-14B")
    # 메모리 해제: base 모델 삭제 후 GPU 캐시 비움
    # → fine-tuned 모델 로딩을 위한 공간 확보
    del base_model
    torch.cuda.empty_cache()

    # === Fine-tuned 모델 평가 ===
    # base 모델을 다시 로드한 뒤, 학습된 LoRA 어댑터를 얹음
    print("\nFine-tuned 모델 로딩 중...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # PeftModel.from_pretrained: 저장된 LoRA 어댑터를 base 모델에 적용
    ft_model = PeftModel.from_pretrained(ft_model, adapter_dir)
    ft_results = evaluate_model(ft_model, tokenizer, val_dataset, config, label="Fine-tuned Qwen3-14B")

    # === 비교 테이블 출력 ===
    print(f"\n{'='*60}")
    print("비교 결과")
    print(f"{'='*60}")
    metrics = ["json_parse_rate", "accuracy", "f1_macro", "prob_mae", "pos_topic_f1", "neg_topic_f1"]
    print(f"{'지표':<20} {'Base':>12} {'Fine-tuned':>12} {'차이':>12}")
    print("-" * 60)
    for m in metrics:
        base_v = base_results[m]
        ft_v = ft_results[m]
        # NaN 체크: prob_mae 등이 NaN일 수 있음 (파싱 실패 시)
        diff = ft_v - base_v if not (base_v != base_v or ft_v != ft_v) else float("nan")
        print(f"{m:<20} {base_v:>12.4f} {ft_v:>12.4f} {diff:>+12.4f}")


if __name__ == "__main__":
    main()
