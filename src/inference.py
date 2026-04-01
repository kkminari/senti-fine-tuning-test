"""
추론 스크립트: LoRA 어댑터를 로딩하여 단일 텍스트 감성 분석
==========================================================
실행: cd src && python inference.py --text "이 제품 정말 좋아요 향도 좋고 발림성도 최고"

하는 일:
  1. base 모델(Qwen3-14B) 4bit 양자화 로드
  2. 학습된 LoRA 어댑터 적용
  3. 입력 텍스트를 프롬프트로 감싸서 모델에 전달
  4. 모델의 JSON 응답을 파싱하여 출력

출력 예시:
  {
    "sentiment": "positive",
    "probability": 0.92,
    "positive_topics": ["향", "발림성"],
    "negative_topics": []
  }

참고 - 추론 시 로딩 구조:
  base 모델 (~7GB, 4bit) + LoRA 어댑터 (~수십MB) = fine-tuned 모델
  → 어댑터만 교체하면 다른 태스크에도 같은 base 모델 재활용 가능
"""

import argparse
import json
import yaml
import torch
from dotenv import load_dotenv

load_dotenv()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from data_loader import SYSTEM_PROMPT, USER_TEMPLATE


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(config: dict):
    """양자화된 base 모델 + LoRA 어댑터 로딩

    추론 시에도 학습 때와 동일한 양자화 설정을 사용해야 함.
    (다른 설정으로 로드하면 가중치 해석이 달라져서 결과가 이상해짐)
    """
    model_name = config["model"]["name"]
    adapter_dir = config["output"]["adapter_dir"]

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # 저장된 LoRA 어댑터를 base 모델 위에 적용
    model = PeftModel.from_pretrained(model, adapter_dir)
    # model.eval(): 드롭아웃 비활성화, BatchNorm을 평가 모드로 전환
    # 학습 때와 추론 때의 동작이 달라지는 레이어들이 있어서 필수
    model.eval()

    return model, tokenizer


def predict(model, tokenizer, text: str, config: dict) -> dict:
    """텍스트 입력 → JSON 감성분석 결과 반환

    프롬프트 구성:
      학습 때 사용한 것과 동일한 system/user 프롬프트를 사용해야 함.
      (다른 프롬프트를 쓰면 모델이 학습한 패턴과 달라져서 성능 저하)

    add_generation_prompt=True:
      학습 때는 False (정답 포함), 추론 때는 True (모델이 생성)
      → assistant 시작 토큰까지만 프롬프트에 포함하고, 나머지는 모델이 채움

    outputs[0][inputs["input_ids"].shape[1]:]:
      모델 출력에서 프롬프트 부분을 잘라내고 생성된 토큰만 디코딩
      (generate()는 프롬프트 + 생성 토큰을 합쳐서 반환하므로)
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
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # JSON 파싱: 모델 출력에서 { } 사이의 JSON 추출
    try:
        if "{" in response and "}" in response:
            start = response.index("{")
            end = response.rindex("}") + 1
            return json.loads(response[start:end])
    except (json.JSONDecodeError, ValueError):
        pass

    # 파싱 실패 시 raw 응답 반환 (디버깅용)
    return {"raw_response": response, "error": "JSON 파싱 실패"}


def main():
    parser = argparse.ArgumentParser(description="한국어 감성 분석 추론")
    parser.add_argument("--text", type=str, required=True, help="분석할 텍스트")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print("모델 로딩 중...")
    model, tokenizer = load_model(config)

    result = predict(model, tokenizer, args.text, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
