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
이 영화 진짜 재밌어요 배우 연기도 좋고 스토리도 탄탄해서 강추합니다
```

**출력**
```json
{
  "sentiment": "positive",
  "probability": 0.95,
  "positive_topics": ["영화", "재미", "배우", "연기", "스토리"],
  "negative_topics": []
}
```

---

## 최종 성능 (v4)

4단계 실험을 거쳐 달성한 최종 성능입니다 (test set 23건 기준).

| 메트릭 | Base 모델 | Fine-tuned (v4) | 변화 |
|--------|-----------|-----------------|------|
| JSON 파싱률 | 100% | **100%** | ±0% |
| 정확도 | 0% | **100%** | +100%p |
| F1 (Macro) | 0% | **100%** | +100%p |
| F1 (Weighted) | 0% | **100%** | +100%p |
| 확률 MAE | 0.227 | **0.044** | -81% |
| 긍정 토픽 F1 | 39.1% | **80.1%** | +41.0%p |
| 부정 토픽 F1 | 55.7% | **91.0%** | +35.3%p |

### 최적화 과정

| 버전 | Epochs | Target Modules | 데이터 분할 | Eval Loss | 정확도 | 토픽 F1 |
|------|--------|---------------|-----------|-----------|--------|---------|
| v1 | 5 | Attention (4개) | train/val | 0.631 | 83.3% | 75~79% |
| v2 | 3 | +MLP (7개) | train/val | 0.496 | 80.0% | 80~84% |
| v3 | 4 | +MLP (7개) | train/val | 0.480 | 83.3% | 82~83% |
| **v4** | **4** | **+MLP (7개)** | **train/val/test** | **0.222** | **100%** | **80~91%** |

---

## Project Structure

```
senti-fine-tuning-test/
├── configs/
│   └── training_config.yaml        # 하이퍼파라미터 설정 (QLoRA, 학습, 추론)
├── src/
│   ├── data_loader.py              # 데이터 로딩, 전처리, 프롬프트 템플릿
│   ├── train.py                    # QLoRA 학습 메인 스크립트
│   ├── evaluate.py                 # Base vs Fine-tuned 비교 평가
│   ├── inference.py                # 단일 텍스트 추론 파이프라인
│   └── cross_validate.py           # K-fold 교차 검증
├── notebooks/
│   ├── eda.ipynb                   # 데이터 탐색 (감성분포, 토픽빈도, 텍스트길이)
│   └── inference_test.ipynb        # 추론 테스트 노트북 (직접 실행용)
├── reports/
│   ├── generate_report.py          # v1 결과 보고서 생성
│   ├── generate_report_v2.py       # v2 비교 보고서 생성
│   ├── generate_report_v3.py       # v3 보고서 생성
│   └── generate_report_v4.py       # v4 종합 보고서 생성 (v1~v4 비교)
├── docs/
│   ├── EXECUTION_PLAN.md           # 실행 계획서 및 진행 추적
│   └── plans/
│       ├── 2026-03-30-senti-finetuning-design.md
│       └── IMPLEMENTATION_TRACKER.md
├── requirements.txt
├── .env                            # HF_TOKEN, WANDB_API_KEY (git 미포함)
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
python src/train.py
```

- 학습 진행 상황은 [WandB 대시보드](https://wandb.ai)에서 실시간 모니터링
- 완료 시 `outputs/adapter/`에 LoRA 어댑터 저장 (~84MB)

### 4. 평가

```bash
PYTHONPATH=src python src/evaluate.py
```

Base 모델과 Fine-tuned 모델의 성능을 비교합니다.

### 5. 추론

```bash
PYTHONPATH=src python src/inference.py --text "이 영화 진짜 재밌어요 배우 연기도 좋고"
```

### 6. 보고서 생성

```bash
python reports/generate_report_v4.py
```

`outputs/finetuning_report_v4.pdf`에 9페이지 종합 분석 보고서가 생성됩니다.

---

## Training Details (v4 최종 설정)

### QLoRA 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Quantization | 4-bit NF4 | 모델 메모리 ~56GB → ~8GB |
| LoRA rank | 16 | 학습 파라미터 수 조절 |
| LoRA alpha | 32 | 스케일링 팩터 (rank의 2배) |
| LoRA dropout | 0.1 | 과적합 방지 (150건 소규모 데이터 대응) |
| Target modules | q, k, v, o, gate, up, down_proj | Attention + MLP (7개 모듈) |
| 학습 가능 파라미터 | 64.2M / 14.8B (0.43%) | |

### 학습 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Epochs | 4 (+ early stopping) | v1(5)→v2(3)→v3(4) 최적화 |
| Effective batch size | 8 | 4 batch x 2 accumulation |
| Learning rate | 2e-4 | QLoRA 표준 (cosine scheduler) |
| Optimizer | Paged AdamW 8bit | 메모리 효율적 |
| Precision | BFloat16 | A100 최적 |
| Attention | SDPA | PyTorch 내장 |

### 데이터 분할 (v4에서 3분할 적용)

| 구분 | 건수 | 비율 | 용도 |
|------|------|------|------|
| Train | 104 | 70% | 모델 학습 |
| Validation | 23 | 15% | 학습 중 과적합 모니터링 |
| Test | 23 | 15% | 최종 성능 측정 (1회만 사용) |

---

## Tech Stack

| 패키지 | 용도 |
|--------|------|
| torch | GPU 연산 및 딥러닝 프레임워크 |
| transformers | 사전학습 모델 로딩 및 학습 |
| peft | LoRA 어댑터 (Parameter-Efficient Fine-Tuning) |
| trl | SFTTrainer (Supervised Fine-Tuning) |
| bitsandbytes | 4bit 양자화 |
| datasets | HuggingFace 데이터셋 로딩 |
| accelerate | 분산 학습 / 디바이스 매핑 |
| wandb | 실험 추적 및 모니터링 |
| scikit-learn | 평가 지표 (Accuracy, F1) |
| matplotlib | PDF 보고서 차트 생성 |

---

## Notes

- 모든 파라미터는 `configs/training_config.yaml`에서 중앙 관리됩니다.
- 코드 내 주석에 각 파라미터의 선택 근거와 핵심 개념이 설명되어 있습니다.
- `.env` 파일은 `.gitignore`에 포함되어 있어 토큰이 원격 저장소에 노출되지 않습니다.
- 4단계 최적화 과정 및 상세 분석은 `reports/generate_report_v4.py`로 생성되는 PDF 보고서를 참고하세요.
