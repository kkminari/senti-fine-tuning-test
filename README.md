# Qwen3-14B 한국어 감성분석 Fine-tuning

Qwen3-14B를 **QLoRA**로 파인튜닝하여, 한국어 텍스트의 감성(긍정/부정/중립)과 토픽을 JSON으로 출력하는 모델을 만드는 프로젝트입니다.

> **목적**: 학습/연구용

---

## Overview

| 항목 | 내용 |
|------|------|
| Base Model | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |
| 파인튜닝 기법 | QLoRA (4-bit NF4 + LoRA r=16) |
| 프레임워크 | Transformers + PEFT + TRL |
| 데이터셋 | [Younggooo/senti_data2](https://huggingface.co/datasets/Younggooo/senti_data2) (150건) |
| 학습 환경 | A100 80GB |
| 실험 추적 | Weights & Biases |

### 모델 입출력 예시

**입력**
```
색상 진짜 이뻐요 잘 지워지지도 않아요 향도 좋고 가볍게 잘 발립니다. 강추
```

**출력**
```json
{
  "sentiment": "positive",
  "probability": 0.95,
  "positive_topics": ["색상", "향", "발림"],
  "negative_topics": []
}
```

---

## Project Structure

```
senti-fine-tuning-test/
├── configs/
│   └── training_config.yaml    # 하이퍼파라미터 설정 (QLoRA, 학습, 추론)
├── src/
│   ├── data_loader.py          # 데이터 로딩, 전처리, 프롬프트 템플릿
│   ├── train.py                # QLoRA 학습 메인 스크립트
│   ├── evaluate.py             # Base vs Fine-tuned 비교 평가
│   └── inference.py            # 단일 텍스트 추론 파이프라인
├── notebooks/
│   └── eda.ipynb               # 데이터 탐색 (감성분포, 토픽빈도, 텍스트길이)
├── docs/plans/
│   ├── 2026-03-30-senti-finetuning-design.md   # 설계 문서
│   └── IMPLEMENTATION_TRACKER.md               # 구현 진행 추적
├── requirements.txt
├── .env                        # HF_TOKEN, WANDB_API_KEY (git 미포함)
└── .gitignore
```

---

## Quick Start

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. API 키 설정

`.env` 파일에 토큰 입력:

```
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

### 3. 학습

```bash
cd src
python train.py
```

- 학습 진행 상황은 [WandB 대시보드](https://wandb.ai)에서 실시간 모니터링
- 완료 시 `outputs/adapter/`에 LoRA 어댑터 저장

### 4. 평가

```bash
python evaluate.py
```

Base 모델과 Fine-tuned 모델의 성능을 비교합니다:

| 지표 | 설명 |
|------|------|
| `json_parse_rate` | 유효한 JSON 출력 비율 |
| `accuracy` | 감성 분류 정확도 |
| `f1_macro` | 클래스별 F1 평균 |
| `prob_mae` | 확률값 예측 오차 (낮을수록 좋음) |
| `pos_topic_f1` | 긍정 토픽 추출 F1 |
| `neg_topic_f1` | 부정 토픽 추출 F1 |

### 5. 추론

```bash
python inference.py --text "이 영화 진짜 재밌어요 배우 연기도 좋고"
```

---

## Training Details

### QLoRA 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Quantization | 4-bit NF4 | 모델 메모리 ~56GB → ~7GB |
| LoRA rank | 16 | 학습 파라미터 수 조절 (전체의 ~0.2%) |
| LoRA alpha | 32 | 스케일링 팩터 (rank의 2배) |
| LoRA dropout | 0.05 | 과적합 방지 |
| Target modules | q, k, v, o_proj | Attention 레이어만 학습 |

### 학습 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Epochs | 5 (+ early stopping) | patience=2 |
| Effective batch size | 16 | 4 batch x 4 accumulation |
| Learning rate | 2e-4 | QLoRA 표준 |
| Scheduler | Cosine | 후반부 안정적 수렴 |
| Optimizer | Paged AdamW 8bit | 메모리 효율적 |
| Precision | BFloat16 | A100 최적 |

### 데이터 분할

| 구분 | 건수 | 비율 |
|------|------|------|
| Train | 120 | 80% |
| Validation | 30 | 20% |

---

## Tech Stack

```
torch           — GPU 연산 및 딥러닝 프레임워크
transformers    — 사전학습 모델 로딩 및 학습
peft            — LoRA 어댑터 (Parameter-Efficient Fine-Tuning)
trl             — SFTTrainer (Supervised Fine-Tuning)
bitsandbytes    — 4bit 양자화
datasets        — HuggingFace 데이터셋 로딩
accelerate      — 분산 학습 / 디바이스 매핑
wandb           — 실험 추적 및 모니터링
scikit-learn    — 평가 지표 (Accuracy, F1)
```

---

## Notes

- 모든 파라미터는 `configs/training_config.yaml`에서 중앙 관리됩니다.
- 코드 내 주석에 각 파라미터의 선택 근거와 핵심 개념이 설명되어 있습니다.
- `.env` 파일은 `.gitignore`에 포함되어 있어 토큰이 원격 저장소에 노출되지 않습니다.
