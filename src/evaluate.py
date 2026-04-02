"""
평가 스크립트: base 모델 vs fine-tuned 모델 비교
================================================
실행: cd src && python evaluate.py

개선 사항 (v4):
  [1-3] 평가 시 do_sample=False (greedy) — 재현 가능한 결과
  [1-4] Qwen3 thinking mode 비활성화 — <think> 태그 제거
  [2-2] Confusion matrix 출력 — 어떤 클래스를 틀리는지 시각화
  [2-3] 오답 분석 — 틀린 샘플 상세 출력
  [2-4] 평가 결과 JSON 저장 — outputs/eval_results.json
  [3-1] Test set 사용 — val이 아닌 test로 최종 평가
"""

import json
import os
import yaml
import torch
from dotenv import load_dotenv

load_dotenv()

from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from data_loader import load_and_split, SYSTEM_PROMPT, USER_TEMPLATE, strip_thinking


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_response(model, tokenizer, text: str, config: dict) -> str:
    """단일 텍스트에 대해 모델 응답 생성

    [1-3] do_sample=False (greedy decoding):
      평가에서는 같은 입력에 항상 같은 출력이 나와야 재현 가능(reproducible).
      do_sample=True + temperature=0.1이면 거의 같지만 "거의"는 과학이 아님.
      greedy는 확률 가장 높은 토큰을 무조건 선택 → 100% 동일 결과.

    [1-4] enable_thinking=False:
      Qwen3의 <think>...</think> 출력을 비활성화.
      + strip_thinking()으로 혹시 나온 thinking도 제거 (이중 안전장치).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # [1-4] thinking 비활성화
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config["inference"]["max_new_tokens"],
            do_sample=False,  # [1-3] greedy decoding — 재현성 확보
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # [1-4] thinking 블록이 혹시 남아있으면 제거
    response = strip_thinking(response)
    return response.strip()


def parse_response(response: str) -> dict | None:
    """모델 응답을 JSON으로 파싱 시도"""
    try:
        if "{" in response and "}" in response:
            start = response.index("{")
            end = response.rindex("}") + 1
            return json.loads(response[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def compute_topic_f1(pred_topics: list[str], gold_topics: list[str]) -> dict:
    """토픽 리스트 간 precision, recall, F1 계산"""
    if not gold_topics and not pred_topics:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gold_topics or not pred_topics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_set = Counter(pred_topics)
    gold_set = Counter(gold_topics)
    common = sum((pred_set & gold_set).values())

    precision = common / sum(pred_set.values()) if pred_set else 0.0
    recall = common / sum(gold_set.values()) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def print_confusion_matrix(gold: list[str], pred: list[str]):
    """[2-2] Confusion matrix 출력

    혼동 행렬이란:
      행 = 실제 정답, 열 = 모델 예측
      대각선이 높을수록 좋고, 대각선 밖은 오분류

    예시:
                  예측
                  positive  negative  neutral
      정답 positive [  14       1        0   ]
           negative [   2       8        0   ]
           neutral  [   1       1        3   ]

    이걸 보면 "부정→긍정 오판 2건"처럼 구체적 약점 파악 가능
    """
    # parse_fail 포함한 모든 라벨
    all_labels = sorted(set(gold + pred))
    # 감성 라벨을 앞에, parse_fail은 뒤에
    sentiment_order = ["positive", "negative", "neutral"]
    labels = [l for l in sentiment_order if l in all_labels] + [l for l in all_labels if l not in sentiment_order]

    cm = confusion_matrix(gold, pred, labels=labels)

    print(f"\n{'='*60}")
    print("[2-2] Confusion Matrix")
    print(f"{'='*60}")
    # 헤더
    header = f"{'정답\\예측':>14}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = f"{label:>14}" + "".join(f"{cm[i][j]:>12}" for j in range(len(labels)))
        print(row)
    print()

    # sklearn classification_report (precision, recall, f1 per class)
    print(classification_report(gold, pred, labels=labels, zero_division=0))


def print_error_analysis(errors: list[dict]):
    """[2-3] 오답 분석 — 틀린 샘플 상세 출력

    틀린 샘플을 직접 보면:
      - 데이터 라벨링이 잘못된 건지
      - 모호한 텍스트라서 모델이 헷갈린 건지
      - 프롬프트/학습이 부족한 건지
    진단이 가능함. 숫자만 보면 절대 알 수 없는 정보.
    """
    if not errors:
        print("\n오답 없음!")
        return

    print(f"\n{'='*60}")
    print(f"[2-3] 오답 분석 ({len(errors)}건)")
    print(f"{'='*60}")
    for i, err in enumerate(errors, 1):
        print(f"\n--- 오답 #{i} ---")
        print(f"  입력: {err['input'][:100]}{'...' if len(err['input']) > 100 else ''}")
        print(f"  정답: {err['gold_sentiment']} (prob: {err['gold_prob']})")
        print(f"  예측: {err['pred_sentiment']} (prob: {err.get('pred_prob', 'N/A')})")
        if err.get("raw_response"):
            print(f"  원본 응답: {err['raw_response'][:150]}")


def evaluate_model(model, tokenizer, dataset, config: dict, label: str = "model") -> dict:
    """데이터셋 전체에 대해 모델 평가 실행"""
    gold_sentiments = []
    pred_sentiments = []
    prob_errors = []
    pos_topic_f1s = []
    neg_topic_f1s = []
    json_success = 0
    total = len(dataset)
    errors = []  # [2-3] 오답 수집
    all_predictions = []  # [2-4] 전체 예측 결과 저장용

    print(f"\n{'='*60}")
    print(f"평가 시작: {label} ({total}건)")
    print(f"{'='*60}")

    for i, example in enumerate(dataset):
        response = generate_response(model, tokenizer, example["Input"], config)
        parsed = parse_response(response)

        prediction = {
            "index": i,
            "input": example["Input"],
            "gold_sentiment": example["sentiment"],
            "gold_probability": float(example["probability"]) if isinstance(example["probability"], str) else example["probability"],
            "raw_response": response,
        }

        if parsed is not None:
            json_success += 1
            gold_sentiments.append(example["sentiment"])
            pred_sentiments.append(parsed.get("sentiment", "unknown"))

            gold_prob = prediction["gold_probability"]
            pred_prob = parsed.get("probability", 0.0)
            if isinstance(pred_prob, (int, float)):
                prob_errors.append(abs(gold_prob - pred_prob))

            gold_pos = [t.strip() for t in (example.get("positive_topics", "") or "").split(",") if t.strip()]
            gold_neg = [t.strip() for t in (example.get("negative_topics", "") or "").split(",") if t.strip()]
            pred_pos = parsed.get("positive_topics", [])
            pred_neg = parsed.get("negative_topics", [])
            if isinstance(pred_pos, list):
                pos_topic_f1s.append(compute_topic_f1(pred_pos, gold_pos)["f1"])
            if isinstance(pred_neg, list):
                neg_topic_f1s.append(compute_topic_f1(pred_neg, gold_neg)["f1"])

            prediction.update({
                "pred_sentiment": parsed.get("sentiment", "unknown"),
                "pred_probability": pred_prob,
                "pred_positive_topics": pred_pos if isinstance(pred_pos, list) else [],
                "pred_negative_topics": pred_neg if isinstance(pred_neg, list) else [],
                "json_parsed": True,
            })

            # [2-3] 오답 수집
            if example["sentiment"] != parsed.get("sentiment"):
                errors.append({
                    "input": example["Input"],
                    "gold_sentiment": example["sentiment"],
                    "gold_prob": gold_prob,
                    "pred_sentiment": parsed.get("sentiment", "unknown"),
                    "pred_prob": pred_prob,
                    "raw_response": response,
                })
        else:
            gold_sentiments.append(example["sentiment"])
            pred_sentiments.append("parse_fail")
            prediction.update({"pred_sentiment": "parse_fail", "json_parsed": False})
            errors.append({
                "input": example["Input"],
                "gold_sentiment": example["sentiment"],
                "gold_prob": prediction["gold_probability"],
                "pred_sentiment": "parse_fail",
                "raw_response": response,
            })

        all_predictions.append(prediction)

        if (i + 1) % 10 == 0:
            print(f"  진행: {i+1}/{total}")

    # 지표 계산
    results = {
        "label": label,
        "total": total,
        "json_parse_rate": json_success / total if total > 0 else 0.0,
        "accuracy": accuracy_score(gold_sentiments, pred_sentiments),
        "f1_macro": f1_score(gold_sentiments, pred_sentiments, average="macro", zero_division=0),
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

    # [2-2] Confusion matrix
    print_confusion_matrix(gold_sentiments, pred_sentiments)

    # [2-3] 오답 분석
    print_error_analysis(errors)

    return results, all_predictions


def main():
    config = load_config()
    model_name = config["model"]["name"]
    adapter_dir = config["output"]["adapter_dir"]

    # [3-1] test set으로 최종 평가 (val이 아닌 test 사용)
    data_cfg = config["data"]
    dataset_dict = load_and_split(
        data_cfg["dataset_name"],
        data_cfg["train_ratio"],
        data_cfg["val_ratio"],
        data_cfg["test_ratio"],
        data_cfg["seed"],
    )
    test_dataset = dataset_dict["test"]
    print(f"평가 대상: test set ({len(test_dataset)}건)")

    # 양자화 설정
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
    print("\nBase 모델 로딩 중...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_results, base_preds = evaluate_model(base_model, tokenizer, test_dataset, config, label="Base Qwen3-14B")
    del base_model
    torch.cuda.empty_cache()

    # === Fine-tuned 모델 평가 ===
    print("\nFine-tuned 모델 로딩 중...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(ft_model, adapter_dir)
    ft_results, ft_preds = evaluate_model(ft_model, tokenizer, test_dataset, config, label="Fine-tuned Qwen3-14B")

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
        diff = ft_v - base_v if not (base_v != base_v or ft_v != ft_v) else float("nan")
        print(f"{m:<20} {base_v:>12.4f} {ft_v:>12.4f} {diff:>+12.4f}")

    # [2-4] 평가 결과 JSON 저장
    output_dir = config["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    eval_output = {
        "base_results": {k: v for k, v in base_results.items() if not (isinstance(v, float) and v != v)},
        "finetuned_results": {k: v for k, v in ft_results.items() if not (isinstance(v, float) and v != v)},
        "finetuned_predictions": ft_preds,
    }
    eval_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[2-4] 평가 결과 저장: {eval_path}")


if __name__ == "__main__":
    main()
