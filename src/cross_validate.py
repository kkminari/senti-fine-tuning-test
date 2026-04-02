"""
K-Fold Cross-Validation 스크립트
================================
실행: cd src && python cross_validate.py

[3-2] 150건 소규모 데이터에서 평가 신뢰도를 높이기 위한 교차 검증

일반 평가 (1회):
  train 105건으로 학습 → test 23건으로 평가 → accuracy 83%
  → 1건 차이로 4%p 변동, "운 좋은 23건"에 의존

Cross-Validation (5회):
  전체 150건을 5등분 → 각각 다른 fold를 test로 사용하여 5번 학습+평가
  → 평균 accuracy 82% ± 3.5%
  → 전체 150건을 평가에 활용, 신뢰구간 제공

주의:
  - 5번 학습하므로 시간이 5배 소요 (~8분)
  - WandB에 각 fold가 별도 run으로 기록됨
"""

import json
import os
import yaml
import torch
import wandb
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from data_loader import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    strip_thinking,
    apply_chat_template,
    build_output_json,
    format_chat_messages,
)


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_response(model, tokenizer, text: str, max_new_tokens: int = 128) -> str:
    """단일 텍스트 추론 (greedy, thinking 비활성화)"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return strip_thinking(response).strip()


def parse_response(response: str) -> dict | None:
    """JSON 파싱 시도"""
    try:
        if "{" in response and "}" in response:
            start = response.index("{")
            end = response.rindex("}") + 1
            return json.loads(response[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def evaluate_fold(model, tokenizer, test_data, max_new_tokens: int = 128) -> dict:
    """한 fold의 test 데이터에 대해 평가"""
    gold = []
    pred = []
    json_success = 0

    for example in test_data:
        response = generate_response(model, tokenizer, example["Input"], max_new_tokens)
        parsed = parse_response(response)
        gold.append(example["sentiment"])
        if parsed is not None:
            json_success += 1
            pred.append(parsed.get("sentiment", "unknown"))
        else:
            pred.append("parse_fail")

    total = len(gold)
    return {
        "json_parse_rate": json_success / total if total > 0 else 0.0,
        "accuracy": accuracy_score(gold, pred),
        "f1_macro": f1_score(gold, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(gold, pred, average="weighted", zero_division=0),
    }


def main():
    config = load_config()
    model_name = config["model"]["name"]
    n_folds = 5

    # 전체 데이터 로드
    ds = load_dataset(config["data"]["dataset_name"], split="train")
    labels = ds["sentiment"]

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 양자화 설정
    qconfig = config["quantization"]
    compute_dtype = getattr(torch, qconfig["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qconfig["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qconfig["bnb_4bit_use_double_quant"],
    )

    # LoRA 설정
    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config["data"]["seed"])
    fold_results = []

    print(f"\n{'='*60}")
    print(f"  {n_folds}-Fold Cross-Validation 시작")
    print(f"  전체 데이터: {len(ds)}건, 각 fold test: ~{len(ds)//n_folds}건")
    print(f"{'='*60}")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(ds)), labels)):
        print(f"\n--- Fold {fold_idx+1}/{n_folds} ---")
        print(f"  Train: {len(train_idx)}건, Test: {len(test_idx)}건")

        # WandB run (fold별)
        wandb.init(
            project=config["wandb"]["project"],
            name=f"cv-fold-{fold_idx+1}",
            config={**config, "fold": fold_idx + 1},
            reinit=True,
        )

        # 데이터 분할 및 포맷팅
        train_data = ds.select(train_idx.tolist())
        test_data = ds.select(test_idx.tolist())

        train_dataset = apply_chat_template(
            DatasetDict({"train": train_data}), tokenizer
        )["train"]

        # 모델 로드 (매 fold마다 fresh 로드)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        # 학습 설정
        train_cfg = config["training"]
        total_steps = (len(train_data) // (train_cfg["per_device_train_batch_size"] * train_cfg["gradient_accumulation_steps"])) * train_cfg["num_train_epochs"]
        warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

        training_args = SFTConfig(
            output_dir=f"./outputs/cv_fold_{fold_idx+1}",
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            warmup_steps=warmup_steps,
            optim=train_cfg["optim"],
            fp16=train_cfg["fp16"],
            bf16=train_cfg["bf16"],
            logging_steps=train_cfg["logging_steps"],
            save_strategy="no",
            report_to="wandb",
            seed=config["data"]["seed"],
            dataset_text_field="text",
            max_length=config["model"]["max_seq_length"],
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        trainer.train()

        # 평가
        model.eval()
        fold_result = evaluate_fold(
            model, tokenizer, test_data,
            max_new_tokens=config["inference"]["max_new_tokens"],
        )
        fold_results.append(fold_result)

        print(f"  Fold {fold_idx+1} 결과:")
        for k, v in fold_result.items():
            print(f"    {k}: {v:.4f}")

        wandb.log({f"fold_{k}": v for k, v in fold_result.items()})
        wandb.finish()

        # 메모리 해제
        del model, trainer
        torch.cuda.empty_cache()

    # === 전체 결과 요약 ===
    print(f"\n{'='*60}")
    print(f"  Cross-Validation 최종 결과 ({n_folds}-Fold)")
    print(f"{'='*60}")
    metrics = ["json_parse_rate", "accuracy", "f1_macro", "f1_weighted"]
    print(f"{'지표':<20} {'평균':>10} {'표준편차':>10} {'최소':>10} {'최대':>10}")
    print("-" * 60)

    summary = {}
    for m in metrics:
        values = [r[m] for r in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        summary[m] = {"mean": float(mean), "std": float(std), "min": float(min(values)), "max": float(max(values))}
        print(f"{m:<20} {mean:>10.4f} {std:>10.4f} {min(values):>10.4f} {max(values):>10.4f}")

    # 결과 저장
    output_path = "outputs/cv_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_folds": n_folds,
            "fold_results": fold_results,
            "summary": summary,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
