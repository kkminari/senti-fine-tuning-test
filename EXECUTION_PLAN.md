# 파인튜닝 실행 계획서

## 프로젝트 개요
- **목표**: Qwen3-14B 기반 한국어 감성분석 모델을 QLoRA로 파인튜닝
- **데이터**: Younggooo/senti_data2 (150개 샘플 → 120 train / 30 val)
- **GPU**: A100 80GB
- **예상 소요시간**: 전체 파이프라인 약 2~3시간

---

## 진행 상황

| 단계 | 작업 | 상태 | 비고 |
|------|------|------|------|
| **Step 1** | **환경 설정** | | |
| 1-1 | .env 파일 생성 (HF_TOKEN, WANDB_API_KEY) | [x] 완료 | 설정 완료 |
| 1-2 | pip install -r requirements.txt | [x] 완료 | 전체 설치 성공 |
| 1-3 | GPU 확인 (nvidia-smi, torch.cuda) | [x] 완료 | A100 80GB, CUDA 13.0 |
| **Step 2** | **데이터 확인** | | |
| 2-1 | 데이터셋 다운로드 테스트 | [x] 완료 | 150개 샘플 확인 |
| 2-2 | 데이터 로더 동작 확인 (분할 검증) | [x] 완료 | 120/30 분할 정상 |
| **Step 3** | **학습 실행** | | |
| 3-1 | python src/train.py 실행 | [x] 완료 | 92초, 5 에폭, loss 2.329→0.535 |
| 3-2 | outputs/adapter/ 어댑터 저장 확인 | [x] 완료 | ~84MB safetensors |
| 3-3 | WandB 학습 로그 확인 | [x] 완료 | wandb.ai/mina_kwak-pmi/senti-fine-tuning |
| **Step 4** | **평가 실행** | | |
| 4-1 | python src/evaluate.py 실행 | [x] 완료 | Base vs Fine-tuned 비교 완료 |
| 4-2 | 6개 메트릭 결과 기록 | [x] 완료 | 아래 결과 테이블 참조 |
| **Step 5** | **추론 테스트** | | |
| 5-1 | python src/inference.py --text "테스트" | [x] 완료 | 정상 동작 확인 |
| 5-2 | JSON 출력 형식 검증 | [x] 완료 | sentiment, probability, topics 정상 |

---

## 주요 설정 (configs/training_config.yaml)

| 항목 | 값 |
|------|-----|
| 모델 | Qwen/Qwen3-14B |
| 양자화 | 4bit NF4 (QLoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Epochs | 5 (early stopping patience=2) |
| 배치 사이즈 | 4 (gradient accumulation 4 → 유효 16) |
| Learning rate | 2.0e-4 (cosine scheduler) |
| Optimizer | paged_adamw_8bit |
| Max seq length | 512 |

---

## 실행 명령어 요약

```bash
# Step 1: 환경 설정
pip install -r requirements.txt
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Step 3: 학습
python src/train.py

# Step 4: 평가
python src/evaluate.py

# Step 5: 추론
python src/inference.py --text "이 영화 진짜 재밌어요 배우 연기도 좋고"
```

---

## 평가 결과 기록

| 메트릭 | Base 모델 | Fine-tuned | 차이 |
|--------|-----------|------------|------|
| JSON 파싱률 | 0.0000 | 1.0000 | +1.0000 |
| 정확도 (Accuracy) | 0.0000 | 0.8333 | +0.8333 |
| F1 (Macro) | 0.0000 | 0.8141 | +0.8141 |
| F1 (Weighted) | 0.0000 | 0.8192 | +0.8192 |
| 확률 MAE | N/A | 0.0670 | - |
| 긍정 토픽 F1 | N/A | 0.7588 | - |
| 부정 토픽 F1 | N/A | 0.7941 | - |

---

## 중단 시 재개 가이드

1. 이 파일의 **진행 상황** 표에서 `[x] 완료` 표시된 마지막 단계를 확인
2. 다음 미완료 단계부터 이어서 실행
3. Step 3 (학습) 중단 시: `outputs/` 폴더에 체크포인트가 저장되어 있으면 이어서 학습 가능
4. Step 4~5는 `outputs/adapter/`가 존재해야 실행 가능
