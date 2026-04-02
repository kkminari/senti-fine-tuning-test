"""
Qwen3-14B QLoRA 파인튜닝 메인 스크립트
=====================================
실행: cd src && python train.py

전체 학습 파이프라인:
  1. YAML 설정 파일 로드
  2. WandB 실험 추적 초기화
  3. 토크나이저 & 데이터 준비
  4. 학습 시퀀스 토큰 길이 확인 (max_seq_length 적절성 검증)
  5. 모델 로드 (4bit 양자화로 메모리 절약)
  6. LoRA 어댑터 부착 (학습 가능 파라미터만 추가)
  7. SFTTrainer로 학습 실행
  8. 학습 완료된 LoRA 어댑터만 저장

핵심 개념 - QLoRA:
  Q (Quantization) + LoRA의 조합
  - Quantization: 모델 가중치를 4bit로 압축 → GPU 메모리 대폭 절감
  - LoRA: 전체 모델은 동결하고, 작은 어댑터 행렬만 학습
  - 결과: 14B 모델을 A100 1장에서 학습 가능 (VRAM ~20GB 사용)
"""

import os
import yaml
import torch
import wandb
from dotenv import load_dotenv

# .env에서 HF_TOKEN(모델 다운로드), WANDB_API_KEY(실험 추적) 로드
load_dotenv()

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from data_loader import prepare_dataset


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_quantization(config: dict) -> BitsAndBytesConfig:
    """4bit 양자화 설정 생성"""
    qconfig = config["quantization"]
    compute_dtype = getattr(torch, qconfig["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=qconfig["load_in_4bit"],
        bnb_4bit_quant_type=qconfig["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qconfig["bnb_4bit_use_double_quant"],
    )


def setup_lora(config: dict) -> LoraConfig:
    """LoRA 어댑터 설정 생성"""
    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )


def check_token_lengths(dataset, tokenizer, max_seq_length: int):
    """[2-1] 학습 시퀀스의 실제 토큰 수를 확인하여 max_seq_length 적절성 검증

    왜 필요한가:
      - max_seq_length보다 긴 시퀀스는 잘림(truncation) → 불완전한 JSON이 정답이 됨
      - max_seq_length보다 훨씬 짧으면 → 불필요한 패딩으로 GPU 메모리 낭비
      - 실제 토큰 수를 확인해야 최적의 max_seq_length를 설정 가능

    출력 예시:
      학습 시퀀스 토큰 수: 최소=85, 최대=203, 평균=142.3
      max_seq_length(512) 초과: 0건 → 적절
      권장 max_seq_length: 256 (최대 토큰의 1.25배, 32 단위 올림)
    """
    lengths = [len(tokenizer.encode(ex["text"])) for ex in dataset]
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = sum(lengths) / len(lengths)
    over_limit = sum(1 for l in lengths if l > max_seq_length)

    print(f"\n{'='*60}")
    print(f"[2-1] 학습 시퀀스 토큰 수 분석")
    print(f"{'='*60}")
    print(f"  최소: {min_len}, 최대: {max_len}, 평균: {avg_len:.1f}")
    print(f"  max_seq_length({max_seq_length}) 초과: {over_limit}건", end="")
    if over_limit > 0:
        print(f" ⚠ {over_limit}건의 데이터가 잘림! max_seq_length 증가 필요")
    else:
        print(" → 적절")

    # 최적 max_seq_length 권장 (최대 토큰의 1.25배, 32 단위 올림)
    recommended = ((int(max_len * 1.25) + 31) // 32) * 32
    if recommended < max_seq_length:
        print(f"  권장 max_seq_length: {recommended} (현재 {max_seq_length}에서 줄이면 메모리/속도 이득)")
    print()

    return {"min": min_len, "max": max_len, "avg": avg_len, "over_limit": over_limit}


def main():
    # === 1. 설정 로드 ===
    config = load_config()
    model_name = config["model"]["name"]
    max_seq_length = config["model"]["max_seq_length"]

    # === 2. WandB 초기화 ===
    wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["run_name"],
        config=config,
    )

    # === 3. 토크나이저 & 데이터 준비 ===
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_cfg = config["data"]
    dataset = prepare_dataset(
        dataset_name=data_cfg["dataset_name"],
        tokenizer=tokenizer,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        seed=data_cfg["seed"],
    )
    print(f"Train: {len(dataset['train'])}건, Val: {len(dataset['validation'])}건, Test: {len(dataset['test'])}건")

    # === 4. [2-1] 토큰 길이 확인 ===
    check_token_lengths(dataset["train"], tokenizer, max_seq_length)

    # === 5. 모델 로드 (4bit 양자화) ===
    bnb_config = setup_quantization(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)

    # === 6. LoRA 적용 ===
    lora_config = setup_lora(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # === 7. 학습 인자 설정 ===
    train_cfg = config["training"]
    output_dir = config["output"]["output_dir"]

    # warmup_ratio → warmup_steps 변환 (trl 1.0에서 warmup_ratio deprecated)
    total_steps = (len(dataset["train"]) // (train_cfg["per_device_train_batch_size"] * train_cfg["gradient_accumulation_steps"])) * train_cfg["num_train_epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        optim=train_cfg["optim"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        report_to=train_cfg["report_to"],
        seed=data_cfg["seed"],
        dataset_text_field="text",
        max_length=max_seq_length,
    )

    # === 8. SFTTrainer 구성 & 학습 ===
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("학습을 시작합니다...")
    trainer.train()

    # === 9. LoRA 어댑터 저장 ===
    adapter_dir = config["output"]["adapter_dir"]
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"어댑터 저장 완료: {adapter_dir}")

    wandb.finish()
    print("학습 완료!")


if __name__ == "__main__":
    main()
