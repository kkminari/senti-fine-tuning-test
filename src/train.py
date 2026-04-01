"""
Qwen3-14B QLoRA 파인튜닝 메인 스크립트
=====================================
실행: cd src && python train.py

전체 학습 파이프라인:
  1. YAML 설정 파일 로드
  2. WandB 실험 추적 초기화
  3. 토크나이저 & 데이터 준비
  4. 모델 로드 (4bit 양자화로 메모리 절약)
  5. LoRA 어댑터 부착 (학습 가능 파라미터만 추가)
  6. SFTTrainer로 학습 실행
  7. 학습 완료된 LoRA 어댑터만 저장

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
# 이 줄이 import보다 먼저 와야 환경변수가 제대로 적용됨
load_dotenv()

from transformers import (
    AutoModelForCausalLM,    # 사전학습된 인과적 언어모델 로더
    AutoTokenizer,           # 텍스트 ↔ 토큰 변환기
    BitsAndBytesConfig,      # 4bit/8bit 양자화 설정
    TrainingArguments,       # 학습 하이퍼파라미터 묶음
    EarlyStoppingCallback,   # 과적합 감지 시 학습 조기 종료
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# peft: Parameter-Efficient Fine-Tuning 라이브러리
#   LoraConfig: LoRA 하이퍼파라미터 정의
#   get_peft_model: 기존 모델에 LoRA 레이어를 삽입
#   prepare_model_for_kbit_training: 양자화된 모델을 학습 가능하게 준비
#     → gradient checkpointing 활성화, 일부 레이어 FP32 변환 등

from trl import SFTTrainer
# trl: Transformer Reinforcement Learning 라이브러리
# SFTTrainer: Supervised Fine-Tuning 전용 Trainer
#   → 일반 Trainer와 차이: 텍스트 데이터를 자동으로 토크나이징하고
#     assistant 응답 부분만 loss를 계산해줌

from data_loader import prepare_dataset


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_quantization(config: dict) -> BitsAndBytesConfig:
    """4bit 양자화 설정 생성

    양자화란?
      FP32(32bit) → 4bit으로 모델 가중치를 압축하는 기법
      메모리: 14B * 4byte = ~56GB (FP32) → 14B * 0.5byte = ~7GB (4bit)

    NF4 (NormalFloat4):
      사전학습된 모델의 가중치가 정규분포를 따른다는 점을 이용
      → 정규분포에 최적화된 양자화 구간을 사용하여 정보 손실 최소화
      → 일반 INT4보다 정확도가 높음

    compute_dtype (bfloat16):
      4bit으로 "저장"만 하고, 실제 연산(행렬곱 등)은 bfloat16으로 복원해서 수행
      → 속도와 정확도의 균형

    double_quant:
      양자화 시 사용되는 상수(quantization constants)도 추가로 양자화
      → 파라미터당 약 0.37bit 추가 절약
    """
    qconfig = config["quantization"]
    compute_dtype = getattr(torch, qconfig["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=qconfig["load_in_4bit"],
        bnb_4bit_quant_type=qconfig["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qconfig["bnb_4bit_use_double_quant"],
    )


def setup_lora(config: dict) -> LoraConfig:
    """LoRA 어댑터 설정 생성

    LoRA (Low-Rank Adaptation) 원리:
      원래 가중치 행렬 W (예: 5120 x 5120)를 직접 수정하지 않고,
      작은 행렬 A (5120 x 16) * B (16 x 5120)를 추가로 학습

      추론 시: output = W*x + (alpha/r) * A*B*x
      → W는 동결(frozen), A*B만 학습됨

    학습 파라미터 수 비교:
      Full fine-tuning: 14B 전체 = 140억 개
      LoRA (r=16, 4개 모듈): 약 2600만 개 (~0.2%)
      → 학습 속도 빠르고, 과적합 위험 적고, 저장 용량 작음
    """
    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )


def main():
    # === 1. 설정 로드 ===
    config = load_config()
    model_name = config["model"]["name"]
    max_seq_length = config["model"]["max_seq_length"]

    # === 2. WandB 초기화 ===
    # wandb.init() 후부터 모든 학습 메트릭이 자동으로 대시보드에 기록됨
    # config 딕셔너리를 넘기면 하이퍼파라미터도 WandB에 저장됨
    wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["run_name"],
        config=config,
    )

    # === 3. 토크나이저 & 데이터 준비 ===
    # trust_remote_code=True: Qwen 모델의 커스텀 코드 실행 허용
    # (HuggingFace에 등록된 모델 중 일부는 자체 토크나이저 코드가 필요)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # pad_token 설정: 배치 학습 시 길이가 다른 시퀀스를 맞추기 위한 패딩 토큰
    # Qwen 계열은 pad_token이 없는 경우가 있어서 eos_token으로 대체
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_cfg = config["data"]
    dataset = prepare_dataset(
        dataset_name=data_cfg["dataset_name"],
        tokenizer=tokenizer,
        train_ratio=data_cfg["train_ratio"],
        seed=data_cfg["seed"],
    )
    print(f"Train: {len(dataset['train'])}건, Validation: {len(dataset['validation'])}건")

    # === 4. 모델 로드 (4bit 양자화) ===
    bnb_config = setup_quantization(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",               # GPU가 여러 장이면 자동으로 분산 배치
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # FlashAttention-2: 메모리 효율적인 어텐션 연산
                                                  # → 메모리 사용량 ↓, 속도 ↑
                                                  # → 미설치 시 이 줄 제거해도 동작함 (느려질 뿐)
    )
    # prepare_model_for_kbit_training:
    #   양자화된 모델을 학습 가능하게 만드는 전처리
    #   - gradient checkpointing 활성화 (메모리 절약)
    #   - LayerNorm 등을 FP32로 유지 (학습 안정성)
    model = prepare_model_for_kbit_training(model)

    # === 5. LoRA 적용 ===
    lora_config = setup_lora(config)
    model = get_peft_model(model, lora_config)
    # 학습 가능 파라미터 수 출력 (전체 대비 비율 확인)
    # 예: trainable params: 26,214,400 || all params: 14,167,347,200 || trainable%: 0.185%
    model.print_trainable_parameters()

    # === 6. 학습 인자 설정 ===
    train_cfg = config["training"]
    output_dir = config["output"]["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
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
    )

    # === 7. SFTTrainer 구성 ===
    # SFTTrainer vs 일반 Trainer:
    #   - dataset_text_field="text": 이 컬럼의 텍스트를 자동 토크나이징
    #   - assistant 부분만 loss 계산 (프롬프트는 loss에서 제외)
    #   - max_seq_length로 자동 truncation
    #
    # EarlyStoppingCallback(patience=2):
    #   validation loss가 2 에폭 연속 개선되지 않으면 학습 중단
    #   → 150건 소규모 데이터에서 과적합 방지의 핵심 장치
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # === 8. 학습 실행 ===
    print("학습을 시작합니다...")
    trainer.train()

    # === 9. LoRA 어댑터 저장 ===
    # 전체 모델(~28GB)이 아닌 LoRA 어댑터(~수십MB)만 저장
    # 추론 시: base 모델 + 이 어댑터를 합쳐서 사용
    # 이점: 저장 공간 절약, 여러 실험의 어댑터를 각각 보관 가능
    adapter_dir = config["output"]["adapter_dir"]
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"어댑터 저장 완료: {adapter_dir}")

    wandb.finish()
    print("학습 완료!")


if __name__ == "__main__":
    main()
