"""
app.py — Giai đoạn 6: Giao diện UI điều khiển thời gian thực
==============================================================

Đóng gói toàn bộ pipeline Mechanistic Interpretability vào ứng dụng
Web UI tương tác bằng Gradio.

Tabs:
  1. 🔍 Activation Explorer: Attention heatmap tương tác
  2. 🎯 Linear Probing: Accuracy per layer + Feature direction
  3. 🧠 Sparse Autoencoder: Training + Feature Inspector
  4. 🎛️ Steering: Can thiệp nhân quả với slider
  5. 🔬 Logit Lens: Vocabulary projection per layer

Yếu tố "vượt xa yêu cầu":
  - Giao diện hoàn chỉnh chạy trên Localhost/Colab
  - Các slider tương tác thay đổi tham số real-time
  - Biểu đồ trực quan tích hợp

Người phụ trách: Toàn nhóm
"""

import os
import sys
import torch
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.model_utils import load_model_and_tokenizer, get_device, set_seed
from utils.viz_utils import (
    plot_attention_heatmap,
    plot_probe_accuracy_per_layer,
    plot_sae_loss_curve,
    plot_sparsity_histogram,
    plot_logit_lens,
    plot_steering_comparison,
)

# ============================================================
# GLOBAL STATE
# ============================================================
set_seed(42)
DEVICE = get_device()
MODEL = None
TOKENIZER = None
FEATURE_DIRECTION = None
TARGET_LAYER = None
SAE_MODEL = None


def ensure_model_loaded():
    """Tải model nếu chưa tải."""
    global MODEL, TOKENIZER
    if MODEL is None:
        MODEL, TOKENIZER = load_model_and_tokenizer("gpt2", DEVICE)
    return MODEL, TOKENIZER


def load_feature_direction():
    """Tải feature direction nếu có."""
    global FEATURE_DIRECTION, TARGET_LAYER
    direction_path = os.path.join("checkpoints", "feature_direction.pt")
    if os.path.exists(direction_path):
        data = torch.load(direction_path, weights_only=False)
        FEATURE_DIRECTION = data["direction"]
        TARGET_LAYER = data["layer"]
        return True
    return False


# ============================================================
# TAB 1: ACTIVATION EXPLORER
# ============================================================
def extract_and_visualize_attention(text: str, layer: int, head: int):
    """Trích xuất và hiển thị attention heatmap."""
    model, tokenizer = ensure_model_loaded()
    
    if not text.strip():
        return None, "⚠️ Vui lòng nhập văn bản."
    
    from extract_hooks import ActivationExtractor
    extractor = ActivationExtractor(model, tokenizer, DEVICE)
    
    tokens = extractor.get_tokens(text)
    residual, attention = extractor.extract_single(text)
    
    if layer not in attention:
        return None, f"⚠️ Layer {layer} không có attention data."
    
    attn_matrix = attention[layer][0, head].numpy()
    fig = plot_attention_heatmap(attn_matrix, tokens, layer=layer, head=head)
    
    # Thông tin bổ sung
    info = f"""### Attention Pattern Info
- **Input**: "{text}"
- **Tokens**: {len(tokens)} tokens
- **Layer**: {layer} | **Head**: {head}
- **Matrix shape**: {attn_matrix.shape}

**Giải thích**: Mỗi ô (i,j) cho biết token ở vị trí i "chú ý" (attend) vào 
token ở vị trí j bao nhiêu. Giá trị càng sáng = attention weight càng cao.
"""
    return fig, info


# ============================================================
# TAB 2: LINEAR PROBING
# ============================================================
def run_probing_visualization():
    """Hiển thị kết quả probing từ checkpoints."""
    probe_path = os.path.join("checkpoints", "probe_results.pt")
    
    if not os.path.exists(probe_path):
        return None, "⚠️ Chưa có kết quả probing. Chạy `python linear_probe.py` trước."
    
    results = torch.load(probe_path, weights_only=False)
    
    fig = plot_probe_accuracy_per_layer(
        results["accuracies"],
        results["f1_scores"],
    )
    
    best_layer = results["best_layer"]
    best_acc = results["best_accuracy"]
    best_f1 = results["best_f1"]
    
    info = f"""### Linear Probe Results
- **Best Layer**: Layer {best_layer}
- **Best Accuracy**: {best_acc:.4f} ({best_acc:.1%})
- **Best F1-Score**: {best_f1:.4f}

#### Accuracy per Layer:
| Layer | Accuracy | F1-Score |
|-------|----------|----------|
"""
    for i, (acc, f1) in enumerate(zip(results["accuracies"], results["f1_scores"])):
        marker = " ★" if i == best_layer else ""
        info += f"| Layer {i} | {acc:.4f}{marker} | {f1:.4f} |\n"
    
    info += f"""
**Giải thích**: Layer có accuracy cao nhất là nơi mô hình "hiểu" sentiment 
rõ nhất. Feature Direction Vector được trích xuất từ layer này.
"""
    return fig, info


# ============================================================
# TAB 3: SPARSE AUTOENCODER
# ============================================================
def show_sae_results():
    """Hiển thị kết quả SAE training."""
    sae_path = os.path.join("checkpoints", "sae_weights.pt")
    analysis_path = os.path.join("checkpoints", "sae_feature_analysis.pt")
    
    if not os.path.exists(sae_path):
        return None, None, "⚠️ Chưa train SAE. Chạy `python train_sae.py` trước."
    
    sae_data = torch.load(sae_path, weights_only=False)
    history = sae_data["history"]
    
    # Loss curve
    fig_loss = plot_sae_loss_curve(
        history["total_loss"],
        history["recon_loss"],
        history["sparsity_loss"],
    )
    
    info = f"""### SAE Training Results
- **Architecture**: {sae_data['d_model']} → {sae_data['d_model'] * sae_data['expansion_factor']} → {sae_data['d_model']}
- **Expansion Factor**: {sae_data['expansion_factor']}×
- **L1 Coefficient (λ)**: {sae_data['l1_coeff']}
- **Target Layer**: Layer {sae_data['best_layer']}
- **Final Total Loss**: {history['total_loss'][-1]:.6f}
- **Final Recon Loss**: {history['recon_loss'][-1]:.6f}
- **Final Sparsity Loss**: {history['sparsity_loss'][-1]:.6f}
"""
    
    fig_sparsity = None
    if os.path.exists(analysis_path):
        analysis = torch.load(analysis_path, weights_only=False)
        info += f"""
### Feature Analysis
- **Total Features**: {analysis['n_features']}
- **Dead Features**: {analysis['n_dead']} ({analysis['dead_rate']:.1%})
- **Alive Features**: {analysis['n_alive']}
- **Mean L0 Sparsity**: {analysis['mean_l0']:.1f} active features per sample
"""
        # Top features
        info += "\n### Top Activated Features\n"
        for feat_idx in analysis['top_feature_indices'][:5]:
            examples = analysis['feature_examples'].get(feat_idx, [])
            info += f"\n**Feature #{feat_idx}**:\n"
            for ex in examples[:2]:
                info += f"- [{ex['activation']:.3f}] \"{ex['text'][:100]}...\"\n"
        
        # Tự động dựng biểu đồ độ thưa trên dữ liệu thực tế
        activation_path = os.path.join("checkpoints", "activations.pt")
        if os.path.exists(activation_path):
            try:
                act_data = torch.load(activation_path, weights_only=False)
                all_activations = act_data["activations"]
                best_layer = sae_data["best_layer"]
                layer_activations = all_activations[:, best_layer, :].to(DEVICE)
                
                from train_sae import SparseAutoencoder
                sae = SparseAutoencoder(
                    d_model=sae_data["d_model"],
                    expansion_factor=sae_data["expansion_factor"],
                    l1_coeff=sae_data["l1_coeff"]
                ).to(DEVICE)
                sae.load_state_dict(sae_data["state_dict"])
                sae.eval()
                
                with torch.no_grad():
                    z = sae.encode(layer_activations).cpu().numpy()
                
                fig_sparsity = plot_sparsity_histogram(z)
            except Exception as e:
                print(f"[Error] Failed to generate sparsity plot: {e}")
    
    return fig_loss, fig_sparsity, info


# ============================================================
# TAB 4: STEERING
# ============================================================
def run_steering(prompt: str, coeff: float, max_tokens: int):
    """Chạy steering với hệ số can thiệp cho trước."""
    model, tokenizer = ensure_model_loaded()
    
    if not prompt.strip():
        return "⚠️ Vui lòng nhập prompt.", None
    
    if not load_feature_direction():
        return "⚠️ Chưa có feature direction. Chạy linear_probe.py trước.", None
    
    from steering import ActivationSteerer
    
    steerer = ActivationSteerer(
        model, tokenizer, DEVICE, TARGET_LAYER, FEATURE_DIRECTION
    )
    
    # Sinh text với steering
    steered_text = steerer.generate_with_steering(
        prompt,
        steering_coeff=coeff,
        max_new_tokens=int(max_tokens),
        temperature=0.7,
    )
    
    # Sinh text gốc để so sánh
    original_text = steerer.generate_with_steering(
        prompt,
        steering_coeff=0.0,
        max_new_tokens=int(max_tokens),
        temperature=0.7,
    )
    
    # So sánh probabilities
    original_probs = steerer.get_next_token_probs(prompt, steering_coeff=0.0)
    steered_probs = steerer.get_next_token_probs(prompt, steering_coeff=coeff)
    
    fig = plot_steering_comparison(
        original_probs, steered_probs,
        steering_coeff=coeff,
    )
    
    result = f"""### Steering Results (c = {coeff:+.1f})

**Target Layer**: {TARGET_LAYER}

---

**Original (c = 0)**:
> {original_text}

---

**Steered (c = {coeff:+.1f})**:
> {steered_text}

---

**Công thức**: `h_new = h_old + ({coeff:+.1f}) × v_feature`

**Giải thích**: 
- c > 0: Ép model sinh text thiên về positive sentiment
- c < 0: Ép model sinh text thiên về negative sentiment
- |c| càng lớn → can thiệp càng mạnh
"""
    return result, fig


def run_steering_sweep(prompt: str, max_tokens: int):
    """Chạy steering sweep qua nhiều mức c."""
    model, tokenizer = ensure_model_loaded()
    
    if not prompt.strip():
        return "⚠️ Vui lòng nhập prompt."
    
    if not load_feature_direction():
        return "⚠️ Chưa có feature direction. Chạy linear_probe.py trước."
    
    from steering import ActivationSteerer
    
    steerer = ActivationSteerer(
        model, tokenizer, DEVICE, TARGET_LAYER, FEATURE_DIRECTION
    )
    
    coefficients = [-10, -5, -2, 0, 2, 5, 10]
    
    result = f"""### Steering Sweep — Prompt: "{prompt}"
**Target Layer**: {TARGET_LAYER}

---

"""
    for c in coefficients:
        text = steerer.generate_with_steering(
            prompt,
            steering_coeff=c,
            max_new_tokens=int(max_tokens),
            temperature=0.7,
        )
        label = "🟢 POSITIVE" if c > 0 else "🔴 NEGATIVE" if c < 0 else "⚪ NEUTRAL"
        result += f"**[c = {c:+d}]** {label}\n> {text}\n\n---\n\n"
    
    return result


# ============================================================
# TAB 5: LOGIT LENS
# ============================================================
def run_logit_lens(text: str, position: int):
    """Chạy Logit Lens analysis."""
    model, tokenizer = ensure_model_loaded()
    
    if not text.strip():
        return None, "⚠️ Vui lòng nhập văn bản."
    
    from logit_lens import LogitLens
    
    lens = LogitLens(model, tokenizer, DEVICE)
    results = lens.analyze(text, position=int(position), top_k=5)
    
    fig = plot_logit_lens(
        results["top_tokens_per_layer"][1:],  # Skip embed layer
        results["top_probs_per_layer"][1:],
        input_token=results["input_token"],
        position=results["position"],
    )
    
    # Build info table
    info = f"""### Logit Lens Analysis
- **Input**: "{text}"
- **Tokens**: {results['tokens']}
- **Position**: {results['position']} (token: "{results['input_token']}")

#### Predictions per Layer:
| Layer | Top-1 Token | Prob | Top-2 | Top-3 |
|-------|-------------|------|-------|-------|
"""
    for layer_idx in range(1, len(results["top_tokens_per_layer"])):
        tokens = results["top_tokens_per_layer"][layer_idx]
        probs = results["top_probs_per_layer"][layer_idx]
        info += f"| Layer {layer_idx-1} | '{tokens[0]}' | {probs[0]:.4f} | '{tokens[1]}' | '{tokens[2]}' |\n"
    
    info += """
**Giải thích**: Bảng cho thấy model "nghĩ gì" ở mỗi tầng xử lý.
- Early layers: Dự đoán chưa rõ ràng
- Middle layers: Bắt đầu hình thành khái niệm
- Final layers: Dự đoán chính xác với confidence cao
"""
    return fig, info


# ============================================================
# BUILD GRADIO APP
# ============================================================
def build_app() -> gr.Blocks:
    """Xây dựng ứng dụng Gradio với 5 tabs."""
    
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
    }
    .tab-nav button {
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    /* Centering and scaling plots */
    .gradio-plot {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        min-height: 500px !important;
    }
    .gradio-plot img, .gradio-plot svg {
        max-width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
    }
    """
    
    with gr.Blocks(
        title="Mechanistic Interpretability Explorer",
        theme=gr.themes.Soft(
            primary_hue="rose",
            secondary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=custom_css,
    ) as app:
        
        gr.Markdown("""
        # 🧠 Mechanistic Interpretability Explorer
        ### Tutorial Demo — GPT-2 Small (124M parameters)
        
        Khám phá cách mô hình ngôn ngữ "suy nghĩ" bên trong thông qua các kỹ thuật
        Mechanistic Interpretability: **Probing**, **SAE**, **Steering**, và **Logit Lens**.
        
        ---
        """)
        
        with gr.Tabs():
            # ==================== TAB 1 ====================
            with gr.Tab("🔍 Attention Explorer"):
                gr.Markdown("""
                ### Attention Pattern Visualization
                *Method 6 trong tutorial — "Visualizing the attention distribution 
                of a head provides observations for further human explanations"*
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        attn_text = gr.Textbox(
                            label="Input Text",
                            value="When Mary and John went to the store, John gave a drink to Mary",
                            lines=2,
                        )
                        with gr.Row():
                            attn_layer = gr.Slider(0, 11, value=0, step=1, label="Layer")
                            attn_head = gr.Slider(0, 11, value=7, step=1, label="Head")
                        attn_btn = gr.Button("🔍 Visualize Attention", variant="primary")
                    
                    with gr.Column(scale=1):
                        attn_info = gr.Markdown()
                
                attn_plot = gr.Plot(label="Attention Heatmap")
                attn_btn.click(
                    extract_and_visualize_attention,
                    inputs=[attn_text, attn_layer, attn_head],
                    outputs=[attn_plot, attn_info],
                )
            
            # ==================== TAB 2 ====================
            with gr.Tab("🎯 Linear Probing"):
                gr.Markdown("""
                ### Linear Probe per Layer
                *Method 1 trong tutorial — "A high probing accuracy on a held-out test set 
                indicates the presence of feature" (Gurnee+23)*
                """)
                
                probe_btn = gr.Button("📊 Show Probing Results", variant="primary")
                probe_plot = gr.Plot(label="Accuracy per Layer")
                probe_info = gr.Markdown()
                
                probe_btn.click(
                    run_probing_visualization,
                    outputs=[probe_plot, probe_info],
                )
            
            # ==================== TAB 3 ====================
            with gr.Tab("🧠 Sparse Autoencoder"):
                gr.Markdown("""
                ### SAE Training & Feature Analysis
                *Method 4 trong tutorial — "Map superposed activations to a higher-dimensional 
                sparse activation with monosemantic neurons" (Bricken+23)*
                
                **Loss**: L(h, h') = ||h - h'||² + λ||z||₁
                """)
                
                sae_btn = gr.Button("🧠 Show SAE Results", variant="primary")
                
                # Stack plots vertically for uniform layout and maximum width
                sae_loss_plot = gr.Plot(label="Loss Curve")
                sae_sparsity_plot = gr.Plot(label="Sparsity Histogram")
                
                sae_info = gr.Markdown()
                
                sae_btn.click(
                    show_sae_results,
                    outputs=[sae_loss_plot, sae_sparsity_plot, sae_info],
                )
            
            # ==================== TAB 4 ====================
            with gr.Tab("🎛️ Steering"):
                gr.Markdown("""
                ### Causal Intervention — Bẻ lái ngữ nghĩa
                *Causal-Intervention Evaluation trong tutorial — "What happens if we manually 
                increase the activation of a feature?"*
                
                **Công thức**: `h_new = h_old + c × v_feature`
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        steer_prompt = gr.Textbox(
                            label="Prompt",
                            value="The movie was",
                            lines=1,
                        )
                        steer_coeff = gr.Slider(
                            -15, 15, value=5, step=0.5,
                            label="Steering Coefficient (c)",
                            info="c > 0: positive sentiment | c < 0: negative sentiment"
                        )
                        steer_tokens = gr.Slider(
                            10, 100, value=50, step=5,
                            label="Max Tokens",
                        )
                        with gr.Row():
                            steer_btn = gr.Button("🎛️ Steer!", variant="primary")
                            sweep_btn = gr.Button("📈 Sweep All", variant="secondary")
                
                steer_result = gr.Markdown()
                steer_plot = gr.Plot(label="Probability Distribution Comparison")
                
                steer_btn.click(
                    run_steering,
                    inputs=[steer_prompt, steer_coeff, steer_tokens],
                    outputs=[steer_result, steer_plot],
                )
                sweep_btn.click(
                    run_steering_sweep,
                    inputs=[steer_prompt, steer_tokens],
                    outputs=[steer_result],
                )
            
            # ==================== TAB 5 ====================
            with gr.Tab("🔬 Logit Lens"):
                gr.Markdown("""
                ### Vocabulary Projection per Layer
                *Method 2 trong tutorial — "Decode the next token predictions encoded 
                in the activation" (nostalgebraist, 2020)*
                
                **Công thức**: LogitLens(h) = LayerNorm(h) × W_U^T
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        lens_text = gr.Textbox(
                            label="Input Text",
                            value="The capital of France is",
                            lines=1,
                        )
                        lens_position = gr.Slider(
                            -10, 0, value=-1, step=1,
                            label="Position (-1 = last token)",
                        )
                        lens_btn = gr.Button("🔬 Analyze", variant="primary")
                
                lens_plot = gr.Plot(label="Logit Lens Visualization")
                lens_info = gr.Markdown()
                
                lens_btn.click(
                    run_logit_lens,
                    inputs=[lens_text, lens_position],
                    outputs=[lens_plot, lens_info],
                )
        
        gr.Markdown("""
        ---
        ### 📚 References
        - Wang+22: *Interpretability in the Wild: A Circuit for IOI in GPT-2 Small*
        - Bricken+23: *Towards Monosemanticity: Decomposing LMs with Dictionary Learning*
        - nostalgebraist (2020): *Interpreting GPT: the Logit Lens*
        - Gurnee+23: *Finding Neurons in a Haystack*
        - Elhage+22: *Toy Models of Superposition*
        
        **Team**: Duy · Kim · Tín · Tuấn · Trường | CSC14005 — Nhập môn Học máy
        """)
    
    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
