# Qwen3 14B 감성분석 파인튜닝 설계 문서

> 작성일: 2026-03-30
> 목적: 학습/연구용

---

## 1. 프로젝트 개요 및 아키텍처

### 목표

Qwen3 14B를 QLoRA로 파인튜닝하여, 한국어 텍스트를 입력받으면 감성(긍/부정/중립) 확률값 + 긍/부정 토픽을 JSON으로 출력하는 모델 생성

### 기술 스택

| 구분 | 선택 |
|------|------|
| Base Model | Qwen/Qwen3-14B |
| 기법 | QLoRA (4bit NF4 + LoRA) |
| 프레임워크 | transformers + peft + trl |
| 데이터 | Younggooo/senti_data2 (150건) |
| GPU | A100 80GB |
| 출력 형식 | JSON |
| 실험 추적 | Weights & Biases (WandB) |

### 프로젝트 파일 구조

```
senti-fine-tuning-test/
├── configs/
│   └── training_config.yaml    # 하이퍼파라미터 설정
├── src/
│   ├── data_loader.py          # 데이터 로딩 및 전처리
│   ├── train.py                # 학습 메인 스크립트
│   ├── inference.py            # 추론 스크립트
│   └── evaluate.py             # 평가 스크립트
├── notebooks/
│   └── eda.ipynb               # 데이터 탐색
├── requirements.txt
└── README.md
```

---

## 2. 데이터 전처리 및 프롬프트 설계

### 데이터셋 정보

- **출처**: Younggooo/senti_data2 (HuggingFace)
- **총 데이터**: 150건
- **언어**: 한국어
- **도메인**: 제품 리뷰 (화장품, 전자제품, 영화, 음식점 등)

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| Input | string | 사용자 리뷰 텍스트 |
| sentiment | string | 긍정(positive), 부정(negative), 중립(neutral) |
| probability | string | 감성 확률 (0.05 ~ 0.97) |
| positive_topics | string | 긍정 토픽 (쉼표 구분) |
| negative_topics | string | 부정 토픽 (쉼표 구분) |

### 데이터 분할

- Train: 120건 (80%)
- Validation: 30건 (20%)

### 프롬프트 템플릿

```
<|system|>
당신은 한국어 감성 분석 전문가입니다. 주어진 텍스트의 감성을 분석하고 결과를 JSON으로 반환하세요.

<|user|>
다음 텍스트의 감성을 분석하세요:
"색상 진짜 이뻐요 잘 지워지지도 않아요 향도 좋고 가볍게 잘 발립니다. 강추"

<|assistant|>
{"sentiment": "positive", "probability": 0.95, "positive_topics": ["색상", "향", "발림"], "negative_topics": []}
```

### 전처리 포인트

- `probability`: 문자열 -> float 변환
- `positive_topics` / `negative_topics`: 쉼표 구분 문자열 -> 리스트 변환
- 빈 토픽(`""`) -> 빈 리스트(`[]`)로 처리
- 토큰 길이: max 512 tokens (입력+출력 합산)
- 학습 시 Assistant 응답 부분만 loss 계산 (SFTTrainer 자동 처리)

---

## 3. QLoRA 설정 및 학습 하이퍼파라미터

### QLoRA 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| quantization | 4bit NF4 | 메모리 효율적 양자화 |
| LoRA rank (r) | 16 | 학습 파라미터 수 조절 |
| LoRA alpha | 32 | 스케일링 팩터 (보통 r의 2배) |
| LoRA dropout | 0.05 | 과적합 방지 |
| target modules | q_proj, k_proj, v_proj, o_proj | Attention 레이어에 LoRA 적용 |

### 학습 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| epochs | 3~5 | 데이터가 적으므로 5 에폭부터 시작 |
| batch size | 4 | A100 80GB에서 여유 |
| gradient accumulation | 4 | 실효 배치사이즈 = 16 |
| learning rate | 2e-4 | QLoRA 표준 |
| lr scheduler | cosine | 학습 후반부 안정적 수렴 |
| warmup ratio | 0.05 | 초기 학습률 안정화 |
| max seq length | 512 | 데이터 길이 기준 충분 |
| optimizer | paged_adamw_8bit | 메모리 효율적 옵티마이저 |

### WandB 실험 추적

- **트래킹 항목**: Training loss, Validation loss, Learning rate, GPU 사용량
- **설정**: `TrainingArguments`에 `report_to="wandb"` 지정
- **사전 준비**: `wandb login`으로 API key 인증 필요
- 프로젝트명: `senti-fine-tuning`

### 과적합 대응 (150건 데이터 주의)

- Validation loss 모니터링 -> early stopping 적용
- LoRA dropout으로 정규화
- 에폭 수 조절이 핵심 레버

---

## 4. 평가 및 추론

### 평가 지표

| 항목 | 지표 | 설명 |
|------|------|------|
| 감성 분류 | Accuracy, F1-score | 긍/부정/중립 3클래스 분류 정확도 |
| 확률값 | MAE (평균절대오차) | 예측 확률과 정답 확률의 차이 |
| 토픽 추출 | Precision, Recall, F1 | 추출한 토픽이 정답과 얼마나 일치하는지 |
| JSON 유효성 | 파싱 성공률 | 출력이 유효한 JSON인지 비율 |

### 평가 방식

- Validation 30건에 대해 추론 후 위 지표 계산
- 파인튜닝 전/후 비교로 성능 향상 확인 (base Qwen3 14B vs fine-tuned)

### 추론 파이프라인

```python
text = "이 영화 진짜 재밌어요 배우 연기도 좋고"
# -> 프롬프트 템플릿에 삽입
# -> 모델 generate
# -> JSON 파싱
# -> {"sentiment": "positive", "probability": 0.92,
#     "positive_topics": ["영화", "연기"], "negative_topics": []}
```

### 추론 설정

- `temperature`: 0.1 (일관된 JSON 출력을 위해 낮게)
- `max_new_tokens`: 128 (JSON 출력은 짧음)
- LoRA 어댑터만 저장/로딩하여 경량 배포

---

## 5. 작업 단계별 실행 계획

### Step 1. 환경 설정

- `requirements.txt` 작성 (transformers, peft, trl, bitsandbytes, datasets, wandb 등)
- HuggingFace 토큰 설정 (younggoo209)
- WandB 로그인 (`wandb login`)

### Step 2. 데이터 탐색 (EDA)

- 데이터셋 로딩 및 기본 통계 확인
- 감성 분포, 토픽 빈도, 텍스트 길이 분포 시각화

### Step 3. 데이터 전처리

- 프롬프트 템플릿 적용하여 학습 데이터 포맷팅
- train/validation 분할 (120/30)
- 토크나이징

### Step 4. 모델 로딩 및 학습

- Qwen3-14B 4bit 양자화 로딩
- LoRA 어댑터 설정 및 부착
- SFTTrainer로 학습 실행
- 체크포인트 저장

### Step 5. 평가

- base 모델 vs fine-tuned 모델 비교 추론
- 감성 정확도, 확률 MAE, 토픽 F1, JSON 유효성 측정

### Step 6. 추론 스크립트

- 단일 텍스트 입력 -> JSON 출력 파이프라인
- LoRA 어댑터 로딩 방식으로 경량 추론
