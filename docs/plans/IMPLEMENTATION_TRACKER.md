# Qwen3 14B 감성분석 파인튜닝 - 구현 작업계획서

> 최종 수정: 2026-03-31
> 상태: 진행중
> 설계 문서: [2026-03-30-senti-finetuning-design.md](./2026-03-30-senti-finetuning-design.md)

---

## 전체 진행률

| Phase | 상태 | 진행률 |
|-------|------|--------|
| Phase 1. 환경 설정 | 완료 | 4/4 |
| Phase 2. 데이터 탐색 (EDA) | 완료 | 4/4 |
| Phase 3. 데이터 전처리 | 완료 | 5/5 |
| Phase 4. 학습 스크립트 | 완료 | 6/6 |
| Phase 5. 평가 스크립트 | 완료 | 5/5 |
| Phase 6. 추론 스크립트 | 완료 | 2/3 |
| **전체** | **거의 완료** | **26/27** |

---

## Phase 1. 환경 설정

> 산출물: `requirements.txt`, `.env`, `configs/training_config.yaml`

- [x] **1-1. requirements.txt 작성**
  - 파일: `requirements.txt`
  - 패키지 목록:
    - `torch>=2.1.0`
    - `transformers>=4.45.0`
    - `peft>=0.13.0`
    - `trl>=0.12.0`
    - `bitsandbytes>=0.44.0`
    - `datasets>=3.0.0`
    - `accelerate>=1.0.0`
    - `wandb>=0.18.0`
    - `scikit-learn`
    - `pandas`
    - `matplotlib`
    - `seaborn`

- [x] **1-2. HuggingFace 토큰 설정**
  - 파일: `.env`
  - 내용: `HF_TOKEN=<younggoo209 계정 토큰>`
  - `.gitignore`에 `.env` 포함 확인

- [x] **1-3. training_config.yaml 작성**
  - 파일: `configs/training_config.yaml`
  - 내용: 모델명, QLoRA 파라미터, 학습 하이퍼파라미터, WandB 설정 통합
  - 설계 문서 섹션 3 참고

- [x] **1-4. 디렉토리 구조 생성**
  - `configs/`, `src/`, `notebooks/` 디렉토리 생성
  - 완료 확인: `tree` 명령어로 구조 검증

### Phase 1 완료 조건
- `pip install -r requirements.txt` 성공
- `.env` 파일에 HF_TOKEN 존재
- `configs/training_config.yaml` 파싱 가능

---

## Phase 2. 데이터 탐색 (EDA)

> 산출물: `notebooks/eda.ipynb`

- [x] **2-1. 데이터셋 로딩**
  - HuggingFace `datasets` 라이브러리로 `Younggooo/senti_data2` 로드
  - HF_TOKEN 환경변수 사용
  - 기본 정보 출력: shape, columns, dtypes

- [x] **2-2. 감성 분포 분석**
  - positive / negative / neutral 비율 확인
  - 막대 차트 시각화

- [x] **2-3. 토픽 분석**
  - positive_topics, negative_topics 빈도 집계
  - 상위 토픽 워드클라우드 또는 막대 차트

- [x] **2-4. 텍스트 길이 분석**
  - Input 컬럼 문자 수 / 토큰 수 분포
  - max_seq_length 512가 적절한지 검증
  - 히스토그램 시각화

### Phase 2 완료 조건
- 데이터 150건 전량 로딩 확인
- 감성 분포, 토픽 빈도, 텍스트 길이 시각화 완료
- max_seq_length 적절성 확인

---

## Phase 3. 데이터 전처리

> 산출물: `src/data_loader.py`

- [x] **3-1. 데이터 로딩 함수 구현**
  - 파일: `src/data_loader.py`
  - 함수: `load_dataset_from_hf(token: str) -> Dataset`
  - HuggingFace에서 데이터셋 다운로드

- [x] **3-2. 전처리 함수 구현**
  - 함수: `preprocess_example(example: dict) -> dict`
  - probability: str -> float 변환
  - positive_topics / negative_topics: 쉼표 구분 str -> list 변환
  - 빈 문자열 -> 빈 리스트 처리

- [x] **3-3. 프롬프트 포맷팅 함수 구현**
  - 함수: `format_to_chat(example: dict) -> dict`
  - Qwen3 chat template 적용 (system/user/assistant)
  - assistant 응답: JSON 형식으로 변환
  - 예시 출력:
    ```json
    {"sentiment": "positive", "probability": 0.95, "positive_topics": ["색상", "향"], "negative_topics": []}
    ```

- [x] **3-4. 데이터 분할 함수 구현**
  - 함수: `split_dataset(dataset, test_size=0.2, seed=42) -> (train, val)`
  - stratified split (감성 라벨 기준 층화 추출)
  - train: 120건, val: 30건

- [x] **3-5. 전체 파이프라인 통합 및 테스트**
  - 함수: `prepare_dataset(config) -> (train_dataset, val_dataset)`
  - 위 함수들을 순차 호출
  - 샘플 1건 출력하여 포맷 검증

### Phase 3 완료 조건
- `prepare_dataset()` 호출 시 train 120건, val 30건 반환
- 샘플 데이터의 프롬프트 포맷이 설계 문서와 일치
- JSON 응답 형식이 올바르게 생성됨

---

## Phase 4. 학습 스크립트

> 산출물: `src/train.py`

- [x] **4-1. config 로딩 구현**
  - 파일: `src/train.py`
  - `configs/training_config.yaml` 파싱
  - argparse로 config 경로 인자 받기

- [x] **4-2. 모델 로딩 구현 (QLoRA)**
  - BitsAndBytesConfig: 4bit NF4, compute_dtype=bfloat16
  - AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", quantization_config=...)
  - AutoTokenizer 로딩 + pad_token 설정

- [x] **4-3. LoRA 어댑터 설정**
  - LoraConfig: r=16, alpha=32, dropout=0.05
  - target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  - get_peft_model() 적용
  - 학습 가능 파라미터 수 출력

- [x] **4-4. TrainingArguments 설정**
  - output_dir, num_train_epochs, per_device_train_batch_size
  - gradient_accumulation_steps=4
  - learning_rate=2e-4, lr_scheduler_type="cosine"
  - warmup_ratio=0.05, optim="paged_adamw_8bit"
  - evaluation_strategy="epoch", save_strategy="epoch"
  - report_to="wandb"
  - load_best_model_at_end=True (val_loss 기준)
  - bf16=True

- [x] **4-5. WandB 초기화**
  - wandb.init(project="senti-fine-tuning", name=실험명)
  - config 정보 자동 기록
  - 환경변수 WANDB_PROJECT 설정

- [x] **4-6. SFTTrainer 구성 및 학습 실행**
  - SFTTrainer 초기화: model, train_dataset, eval_dataset, tokenizer
  - max_seq_length=512
  - trainer.train() 실행
  - 학습 완료 후 LoRA 어댑터 저장: `outputs/lora_adapter/`
  - wandb.finish() 호출

### Phase 4 완료 조건
- `python src/train.py --config configs/training_config.yaml` 실행 가능
- WandB 대시보드에서 loss 그래프 확인 가능
- `outputs/lora_adapter/` 에 어댑터 파일 저장 완료
- 학습 로그에 train_loss, eval_loss 출력

---

## Phase 5. 평가 스크립트

> 산출물: `src/evaluate.py`

- [x] **5-1. 모델 로딩 (추론 모드)**
  - 파일: `src/evaluate.py`
  - base model + LoRA 어댑터 병합 로딩
  - argparse: --adapter_path, --config

- [x] **5-2. 추론 함수 구현**
  - 함수: `predict(model, tokenizer, text: str) -> dict`
  - 프롬프트 포맷팅 -> generate -> JSON 파싱
  - temperature=0.1, max_new_tokens=128
  - JSON 파싱 실패 시 에러 핸들링

- [x] **5-3. 평가 지표 계산 함수**
  - 함수: `evaluate_all(predictions, ground_truths) -> dict`
  - 감성 분류: Accuracy, macro F1-score (sklearn)
  - 확률값: MAE
  - 토픽 추출: set 기반 Precision, Recall, F1
  - JSON 유효성: 파싱 성공률

- [x] **5-4. Validation set 전체 평가 실행**
  - val 30건 전체 추론
  - 지표 계산 및 결과 출력
  - 결과를 `outputs/eval_results.json`에 저장

- [x] **5-5. base 모델 vs fine-tuned 비교 (선택)**
  - 파인튜닝 전 base Qwen3-14B로 같은 평가 실행
  - 전/후 지표 비교 테이블 출력

### Phase 5 완료 조건
- `python src/evaluate.py --adapter_path outputs/lora_adapter/` 실행 가능
- 4개 지표 (Accuracy, MAE, Topic F1, JSON 유효성) 출력
- `outputs/eval_results.json` 저장 완료

---

## Phase 6. 추론 스크립트

> 산출물: `src/inference.py`

- [x] **6-1. 단일 텍스트 추론 함수**
  - 파일: `src/inference.py`
  - 함수: `run_inference(text: str, adapter_path: str) -> dict`
  - 모델 로딩 -> 프롬프트 생성 -> generate -> JSON 파싱 -> 결과 반환

- [x] **6-2. CLI 인터페이스**
  - argparse: --text, --adapter_path
  - 실행 예시: `python src/inference.py --text "이 제품 좋아요" --adapter_path outputs/lora_adapter/`
  - 결과를 stdout에 JSON 형식으로 출력

- [ ] **6-3. 배치 추론 지원**
  - argparse: --input_file (텍스트 리스트 파일)
  - 결과를 `outputs/predictions.json`에 저장

### Phase 6 완료 조건
- 단일 텍스트 추론 정상 동작
- 배치 추론 결과 파일 생성 확인

---

## 작업 재개 가이드

작업이 중단된 경우, 아래 순서로 재개합니다:

1. **이 파일의 체크박스 확인** — 마지막으로 완료된 태스크 식별
2. **미완료 태스크의 산출물 파일 존재 여부 확인** — 부분 구현된 코드가 있는지 체크
3. **해당 Phase의 완료 조건 확인** — 이전 Phase가 완료 조건을 충족하는지 검증
4. **다음 미완료 태스크부터 순차 진행**

### 파일별 의존 관계

```
requirements.txt          (독립)
configs/training_config.yaml  (독립)
src/data_loader.py        (독립)
src/train.py              (depends: data_loader.py, training_config.yaml)
src/evaluate.py           (depends: data_loader.py, train.py 산출물)
src/inference.py          (depends: data_loader.py, train.py 산출물)
```

### 환경 복원 체크리스트

- [ ] `pip install -r requirements.txt` 실행
- [ ] `.env` 파일에 HF_TOKEN 설정
- [ ] `wandb login` 실행
- [ ] GPU 환경 확인: `python -c "import torch; print(torch.cuda.get_device_name())"`
