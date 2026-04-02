"""
추론 스크립트: LoRA 어댑터를 로딩하여 단일 텍스트 감성 분석
==========================================================
실행: cd src && python inference.py --text "이 제품 정말 좋아요 향도 좋고 발림성도 최고"

개선 사항 (v4):
  [1-4] Qwen3 thinking mode 비활성화 + strip_thinking 이중 안전장치
"""

import argparse
import json
import yaml
import torch
from dotenv import load_dotenv

load_dotenv()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from data_loader import SYSTEM_PROMPT, USER_TEMPLATE, strip_thinking


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(config: dict):
    """양자화된 base 모델 + LoRA 어댑터 로딩"""
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
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    return model, tokenizer


def predict(model, tokenizer, text: str, config: dict) -> dict:
    """텍스트 입력 → JSON 감성분석 결과 반환"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # [1-4] Qwen3 thinking 비활성화
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config["inference"]["max_new_tokens"],
            do_sample=False,  # 추론에서도 greedy로 일관된 결과
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # [1-4] thinking 블록 제거 (이중 안전장치)
    response = strip_thinking(response)

    try:
        if "{" in response and "}" in response:
            start = response.index("{")
            end = response.rindex("}") + 1
            return json.loads(response[start:end])
    except (json.JSONDecodeError, ValueError):
        pass

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
