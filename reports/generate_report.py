"""파인튜닝 결과 보고서 PDF 생성"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np
import os

# 한국어 폰트 설정
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = "outputs/finetuning_report.pdf"


def title_page(fig):
    fig.text(0.5, 0.55, "Qwen3-14B QLoRA 파인튜닝", ha="center", va="center", fontsize=28, fontweight="bold")
    fig.text(0.5, 0.45, "한국어 감성 분석 모델 결과 보고서", ha="center", va="center", fontsize=20, color="#555555")
    fig.text(0.5, 0.32, "2026-04-01", ha="center", va="center", fontsize=14, color="#888888")
    fig.text(0.5, 0.22, "Base Model: Qwen/Qwen3-14B  |  Method: QLoRA (4-bit NF4)  |  Data: 150 samples",
             ha="center", va="center", fontsize=11, color="#888888")


def overview_page(fig):
    fig.text(0.5, 0.92, "1. 프로젝트 개요", ha="center", fontsize=22, fontweight="bold")

    content = [
        ("목표", "한국어 텍스트에 대한 감성 분석 (긍정/부정/중립) + 토픽 추출"),
        ("Base 모델", "Qwen/Qwen3-14B (14B 파라미터)"),
        ("파인튜닝 기법", "QLoRA — 4-bit NF4 양자화 + LoRA (rank=16, alpha=32)"),
        ("학습 가능 파라미터", "20,971,520 / 14,789,278,720 (0.14%)"),
        ("데이터셋", "Younggooo/senti_data2 — 150개 샘플 (Train 120 / Val 30)"),
        ("GPU", "NVIDIA A100 80GB SXM4"),
        ("학습 시간", "약 92초 (5 에폭, 40 스텝)"),
        ("Optimizer", "paged_adamw_8bit (lr=2e-4, cosine scheduler)"),
        ("배치 사이즈", "4 × gradient accumulation 4 = 유효 16"),
        ("출력 형식", 'JSON {"sentiment", "probability", "positive_topics", "negative_topics"}'),
    ]

    y = 0.82
    for label, value in content:
        fig.text(0.08, y, f"  {label}:", fontsize=12, fontweight="bold", color="#333333")
        fig.text(0.35, y, value, fontsize=12, color="#555555")
        y -= 0.055


def training_curve_page(fig):
    fig.text(0.5, 0.95, "2. 학습 곡선", ha="center", fontsize=22, fontweight="bold")

    train_loss = [2.329, 1.298, 0.733, 0.578, 0.535]
    eval_loss = [1.125, 0.715, 0.649, 0.632, 0.631]
    epochs = [1, 2, 3, 4, 5]

    # Loss curve
    ax1 = fig.add_axes([0.1, 0.48, 0.8, 0.4])
    ax1.plot(epochs, train_loss, "o-", color="#2196F3", linewidth=2.5, markersize=8, label="Train Loss")
    ax1.plot(epochs, eval_loss, "s-", color="#FF5722", linewidth=2.5, markersize=8, label="Eval Loss")
    # 과적합 구간 표시
    ax1.axvspan(3, 5, alpha=0.1, color="red", label="과적합 구간")
    ax1.annotate("과적합 시작", xy=(3, 0.649), xytext=(3.5, 1.2),
                 fontsize=10, color="red", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="red"))
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Evaluation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    # Token accuracy
    token_acc = [0.631, 0.836, 0.850, 0.879, 0.883]
    ax2 = fig.add_axes([0.1, 0.06, 0.8, 0.32])
    ax2.bar(epochs, [a * 100 for a in token_acc], color="#4CAF50", alpha=0.8, width=0.6)
    for i, v in enumerate(token_acc):
        ax2.text(epochs[i], v * 100 + 1, f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Token Accuracy (%)", fontsize=12)
    ax2.set_title("Mean Token Accuracy per Epoch", fontsize=14, fontweight="bold")
    ax2.set_xticks(epochs)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")


def comparison_page(fig):
    fig.text(0.5, 0.95, "3. Base vs Fine-tuned 성능 비교", ha="center", fontsize=22, fontweight="bold")

    metrics = ["JSON\n파싱률", "정확도", "F1\n(Macro)", "F1\n(Weighted)", "긍정토픽\nF1", "부정토픽\nF1"]
    base_vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ft_vals = [1.0, 0.8333, 0.8141, 0.8192, 0.7588, 0.7941]

    # 그래프를 before/after 스타일로 변경
    ax = fig.add_axes([0.08, 0.45, 0.84, 0.42])
    x = np.arange(len(metrics))
    width = 0.35

    # Base 모델: 빗금 패턴 + 빨간 테두리로 "0%" 강조
    bars1 = ax.bar(x - width / 2, [max(v * 100, 2) for v in base_vals], width,
                   label="Base Qwen3-14B (ALL 0%)", color="#FFCDD2", edgecolor="#E53935",
                   linewidth=1.5, hatch="///")
    bars2 = ax.bar(x + width / 2, [v * 100 for v in ft_vals], width,
                   label="Fine-tuned", color="#2196F3", edgecolor="#1565C0", linewidth=0.5)

    # Base 모델 값 표시
    for i, bar in enumerate(bars1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                "0%", ha="center", fontsize=9, fontweight="bold", color="#E53935")

    # Fine-tuned 값 표시
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold", color="#1565C0")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("메트릭 비교 — Base 모델은 JSON 출력 자체를 실패하여 전 메트릭 0%",
                 fontsize=12, fontweight="bold")

    # 화살표로 개선폭 강조
    for i in range(len(metrics)):
        if ft_vals[i] > 0:
            ax.annotate("", xy=(x[i] + width / 2, ft_vals[i] * 100 - 2),
                        xytext=(x[i] - width / 2, 4),
                        arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1.5))

    # Table
    col_labels = ["메트릭", "Base 모델", "Fine-tuned", "변화"]
    metrics_flat = ["JSON 파싱률", "정확도", "F1 (Macro)", "F1 (Weighted)", "긍정토픽 F1", "부정토픽 F1"]
    table_data = []
    for i, m in enumerate(metrics_flat):
        b = base_vals[i]
        f = ft_vals[i]
        table_data.append([m, f"{b:.4f}", f"{f:.4f}", f"+{f - b:.4f}"])
    table_data.append(["확률 MAE", "N/A", "0.0670", "-"])

    ax2 = fig.add_axes([0.08, 0.03, 0.84, 0.32])
    ax2.axis("off")
    table = ax2.table(cellText=table_data, colLabels=col_labels, loc="center",
                      cellLoc="center", colColours=["#E3F2FD"] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        if key[0] == 0:
            cell.set_text_props(fontweight="bold")


def inference_page(fig):
    fig.text(0.5, 0.95, "4. 추론 예시", ha="center", fontsize=22, fontweight="bold")

    examples = [
        {
            "input": "이 영화 진짜 재밌어요 배우 연기도 좋고 스토리도 탄탄해서 강추합니다",
            "output": {
                "sentiment": "positive",
                "probability": 0.95,
                "positive_topics": ["영화", "재미", "배우", "연기", "스토리"],
                "negative_topics": [],
            },
        },
    ]

    y = 0.82
    for i, ex in enumerate(examples):
        fig.text(0.08, y, f"입력:", fontsize=13, fontweight="bold", color="#1565C0")
        y -= 0.04
        fig.text(0.08, y, f'"{ex["input"]}"', fontsize=12, color="#333333", style="italic")
        y -= 0.05
        fig.text(0.08, y, "출력:", fontsize=13, fontweight="bold", color="#2E7D32")
        y -= 0.04
        fig.text(0.08, y, f'  sentiment: {ex["output"]["sentiment"]}', fontsize=12, color="#555555", family="Noto Sans CJK JP")
        y -= 0.035
        fig.text(0.08, y, f'  probability: {ex["output"]["probability"]}', fontsize=12, color="#555555", family="Noto Sans CJK JP")
        y -= 0.035
        fig.text(0.08, y, f'  positive_topics: {ex["output"]["positive_topics"]}', fontsize=12, color="#555555", family="Noto Sans CJK JP")
        y -= 0.035
        fig.text(0.08, y, f'  negative_topics: {ex["output"]["negative_topics"]}', fontsize=12, color="#555555", family="Noto Sans CJK JP")
        y -= 0.06

    # Configuration summary
    y -= 0.02
    fig.text(0.5, y, "5. 학습 설정 요약", ha="center", fontsize=22, fontweight="bold")
    y -= 0.06

    config_data = [
        ["항목", "값"],
        ["양자화", "4-bit NF4 (bfloat16 compute)"],
        ["LoRA Rank / Alpha", "16 / 32"],
        ["LoRA Target Modules", "q_proj, k_proj, v_proj, o_proj"],
        ["LoRA Dropout", "0.05"],
        ["Epochs", "5 (Early Stopping patience=2)"],
        ["Learning Rate", "2.0e-4 (cosine scheduler)"],
        ["Batch Size", "4 (× grad accum 4 = 유효 16)"],
        ["Max Seq Length", "512"],
        ["Precision", "bfloat16"],
        ["Attention", "SDPA (PyTorch native)"],
        ["Inference Temp", "0.1"],
    ]

    ax = fig.add_axes([0.08, 0.02, 0.84, y - 0.02])
    ax.axis("off")
    table = ax.table(cellText=config_data[1:], colLabels=config_data[0], loc="center",
                     cellLoc="center", colColours=["#E8F5E9"] * 2)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        if key[0] == 0:
            cell.set_text_props(fontweight="bold")
        if key[1] == 0 and key[0] > 0:
            cell.set_text_props(fontweight="bold")


def review_page(fig):
    """설정 검토: 잘된 부분 + 개선 필요 부분"""
    fig.text(0.5, 0.95, "6. 설정 검토", ha="center", fontsize=22, fontweight="bold")

    # 잘 설정된 부분
    fig.text(0.08, 0.87, "잘 설정된 부분", fontsize=16, fontweight="bold", color="#2E7D32")

    good_items = [
        ("양자화 (NF4 + double quant)", "14B 모델을 ~8GB로 압축. 표준적이고 안정적인 설정."),
        ("LoRA rank=16, alpha=32", "150건 소규모 데이터에 적절한 크기. alpha/r=2.0 관례 준수."),
        ("lr=2e-4 + cosine scheduler", "QLoRA 파인튜닝의 표준 학습률. 안정적인 수렴 확인."),
        ("bf16 + paged_adamw_8bit", "A100 GPU에 최적화. 메모리 효율적인 조합."),
        ("temperature=0.1", "JSON 출력 일관성 확보. 분석 태스크에 적합한 설정."),
    ]

    y = 0.82
    for title, desc in good_items:
        fig.text(0.10, y, f"  {title}", fontsize=11, fontweight="bold", color="#333333")
        fig.text(0.10, y - 0.03, f"    → {desc}", fontsize=10, color="#666666")
        y -= 0.065

    # 개선 필요 부분
    y -= 0.02
    fig.text(0.08, y, "개선이 필요한 부분", fontsize=16, fontweight="bold", color="#E53935")
    y -= 0.05

    fix_items = [
        ("Epochs: 5 → 3", "Epoch 3 이후 eval_loss 정체 (0.649→0.631). train_loss만 감소하여 과적합 시작."),
        ("target_modules: 4개 → 7개", "Attention만 학습 시 분류 성능 제한. MLP(gate/up/down_proj) 추가로 판단 능력 향상."),
        ("gradient_accumulation: 4 → 2", "유효 배치 16→8. 에폭당 스텝 7.5→15로 증가하여 더 세밀한 학습 가능."),
        ("lora_dropout: 0.05 → 0.1", "150건 소규모 데이터에서 과적합 방지를 위해 드롭아웃 강화 필요."),
    ]

    for title, desc in fix_items:
        fig.text(0.10, y, f"  {title}", fontsize=11, fontweight="bold", color="#333333")
        fig.text(0.10, y - 0.03, f"    → {desc}", fontsize=10, color="#666666")
        y -= 0.065

    # 변경 요약 테이블
    y -= 0.02
    ax = fig.add_axes([0.08, 0.02, 0.84, y - 0.02])
    ax.axis("off")
    col_labels = ["항목", "변경 전", "변경 후", "사유"]
    table_data = [
        ["Epochs", "5", "3", "과적합 방지"],
        ["Target Modules", "4개 (Attention)", "7개 (+MLP)", "분류 성능 향상"],
        ["Grad Accumulation", "4 (유효 16)", "2 (유효 8)", "스텝 수 2배 확보"],
        ["LoRA Dropout", "0.05", "0.1", "과적합 방지 강화"],
    ]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
                     cellLoc="center", colColours=["#FFF3E0"] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        if key[0] == 0:
            cell.set_text_props(fontweight="bold")


def conclusion_page(fig):
    fig.text(0.5, 0.92, "7. 결론 및 핵심 성과", ha="center", fontsize=22, fontweight="bold")

    conclusions = [
        ("JSON 형식 준수", "Base 모델 0% → Fine-tuned 100%. 파인튜닝을 통해 구조화된 출력 생성 능력을 완전히 확보."),
        ("감성 분류 정확도", "83.3% 정확도, F1 81.4% 달성. 150개 소규모 데이터에서도 유의미한 성능."),
        ("토픽 추출", "긍정 토픽 F1 75.9%, 부정 토픽 F1 79.4%. 텍스트에서 핵심 토픽을 효과적으로 추출."),
        ("확률 예측", "MAE 0.067. 감성 확률 예측이 실제 값과 평균 6.7%p 이내로 정확."),
        ("효율성", "전체 14B 파라미터 중 0.14%만 학습. 어댑터 크기 ~84MB로 경량 배포 가능."),
        ("학습 속도", "A100 80GB에서 92초 만에 5 에폭 완료. QLoRA 덕분에 빠른 실험 반복 가능."),
    ]

    y = 0.82
    for i, (title, desc) in enumerate(conclusions):
        fig.text(0.08, y, f"  {i+1}. {title}", fontsize=13, fontweight="bold", color="#1565C0")
        y -= 0.04
        fig.text(0.12, y, desc, fontsize=11, color="#555555", wrap=True,
                 transform=fig.transFigure, verticalalignment="top")
        y -= 0.065

    y -= 0.03
    fig.text(0.5, y, "향후 개선 방향", ha="center", fontsize=18, fontweight="bold", color="#333333")
    y -= 0.05

    improvements = [
        "설정 개선 적용: epochs 3, MLP 모듈 추가, dropout 0.1 → 87~90% 정확도 기대",
        "데이터 확장: 150개 → 1,000개 이상 확보 시 정확도 90%+ 기대",
        "LoRA rank 조정: rank=32 또는 64로 실험하여 최적 파라미터 탐색",
        "배치 추론 지원: --input_file 옵션 구현으로 대량 분석 가능",
    ]
    for imp in improvements:
        fig.text(0.12, y, f"•  {imp}", fontsize=11, color="#555555")
        y -= 0.045


def main():
    os.makedirs("outputs", exist_ok=True)

    with PdfPages(OUTPUT_PATH) as pdf:
        pages = [title_page, overview_page, training_curve_page, comparison_page,
                 inference_page, review_page, conclusion_page]
        for page_fn in pages:
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            fig.patch.set_facecolor("white")
            page_fn(fig)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"보고서 생성 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
