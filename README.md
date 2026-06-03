# 🧠 Mechanistic Interpretability for Language Models — Tutorial Demo

Demo code cho tutorial **"A Practical Review of Mechanistic Interpretability for Language Models"**, thực hiện các kỹ thuật MI trên **GPT-2 Small (124M)**.

## 📋 Mục lục

| Giai đoạn | File | Mô tả | Tutorial Method |
|-----------|------|--------|-----------------|
| 1 | `extract_hooks.py` | Trích xuất Activation & Attention | Feature Study (Setup) |
| 2 | `linear_probe.py` | Linear Probing per layer | Method 1: Probing |
| 3 | `train_sae.py` | Sparse Autoencoder | Method 4: SAE |
| 4 | `steering.py` | Can thiệp nhân quả | Causal Intervention |
| 5 | `logit_lens.py` | Vocabulary Projection | Method 2: Vocab Projection |
| 6 | `app.py` | Gradio Web UI | Interactive Demo |

## 🚀 Cài đặt

```bash
# Clone repository
git clone <repo_url>
cd ML-Lab3

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## 🏃 Chạy Pipeline

Chạy tuần tự từng bước:

```bash
# Bước 1: Trích xuất activations (≈5-10 phút)
python extract_hooks.py --max_samples 500

# Bước 2: Train Linear Probes (≈2-3 phút)
python linear_probe.py --epochs 100

# Bước 3: Train SAE (≈5-10 phút)
python train_sae.py --epochs 50 --expansion_factor 8

# Bước 4: Demo Steering
python steering.py --prompt "The movie was"

# Bước 5: Demo Logit Lens
python logit_lens.py --prompt "The capital of France is"

# Bước 6: Khởi chạy Web UI
python app.py
# → Mở browser tại http://localhost:7860
```

## 📁 Cấu trúc thư mục

```
ML-Lab3/
├── README.md               # Tài liệu hướng dẫn
├── requirements.txt        # Dependencies
├── extract_hooks.py        # GĐ 1: Trích xuất Activation
├── linear_probe.py         # GĐ 2: Linear Probing
├── train_sae.py            # GĐ 3: Sparse Autoencoder
├── steering.py             # GĐ 4: Can thiệp nhân quả
├── logit_lens.py           # GĐ 5: Vocabulary Projection
├── app.py                  # GĐ 6: Gradio Web UI
├── utils/                  # Shared utilities
│   ├── __init__.py
│   ├── model_utils.py      # Model loading, device management
│   └── viz_utils.py        # Visualization functions
├── checkpoints/            # Model weights & cached data (auto-generated)
│   ├── activations.pt
│   ├── probe_results.pt
│   ├── feature_direction.pt
│   ├── sae_weights.pt
│   └── sae_feature_analysis.pt
└── outputs/                # Figures & results (auto-generated)
    ├── attention_L*_H*.png
    ├── probe_accuracy_per_layer.png
    ├── sae_loss_curve.png
    ├── sae_sparsity_histogram.png
    ├── steering_positive.png
    ├── steering_negative.png
    └── logit_lens_*.png
```

## 🔬 Chi tiết kỹ thuật

### Mô hình: GPT-2 Small
- **Tham số**: 124M
- **Kiến trúc**: 12 layers, 12 attention heads, d_model = 768
- **Lý do chọn**: Benchmark chuẩn trong nghiên cứu MI (Wang+22, Conmy+23, Bricken+23)

### Dataset: SST-2 (Stanford Sentiment Treebank)
- **Task**: Binary sentiment classification (positive/negative)
- **Size**: 1000 samples (500 pos + 500 neg)
- **Dùng cho**: Linear Probing và SAE training

### Pipeline Flow
```
Input Text → GPT-2 → [Forward Hooks] → Activations
                                            ↓
                                    Linear Probe per Layer
                                            ↓
                                    Best Layer → SAE Training
                                            ↓
                                    Feature Direction Vector
                                            ↓
                                    Steering: h_new = h_old + c × v
```

## 📚 Tài liệu tham khảo

1. Wang+22: *Interpretability in the Wild: A Circuit for IOI in GPT-2 Small* (ICLR 2022)
2. Bricken+23: *Towards Monosemanticity: Decomposing LMs with Dictionary Learning*
3. nostalgebraist (2020): *Interpreting GPT: the Logit Lens*
4. Gurnee+23: *Finding Neurons in a Haystack* (TMLR 2023)
5. Elhage+22: *Toy Models of Superposition*
6. Belrose+23: *Eliciting Latent Predictions with the Tuned Lens*
7. Conmy+23: *Towards Automated Circuit Discovery for Mechanistic Interpretability* (NeurIPS 2023)

## 👥 Phân công

| Thành viên | Phần đóng góp |
|------------|---------------|
| Duy | GĐ 1: Trích xuất Activation + Attention Visualization |
| Kim | GĐ 2: Linear Probing + Feature Direction |
| Tín | GĐ 3: Sparse Autoencoder + Feature Analysis |
| Tuấn | GĐ 4: Steering / Causal Intervention |
| Cả nhóm | GĐ 5-6: Logit Lens + Web UI |
