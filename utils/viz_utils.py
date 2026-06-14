"""
viz_utils.py — Shared visualization utilities.

Cung cấp các hàm vẽ biểu đồ dùng chung cho toàn bộ pipeline:
- Attention heatmap
- Layer-wise accuracy bar chart
- Sparsity histogram
- Logit lens heatmap
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import torch
from typing import List, Optional

# Sử dụng non-interactive backend khi không có display
matplotlib.use("Agg")

# Thiết lập style chung cho tất cả biểu đồ
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "figure.dpi": 120,
    "font.size": 11,
})


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: List[str],
    layer: int,
    head: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ attention heatmap cho một attention head cụ thể.
    
    Hiển thị ma trận attention (token x token) cho thấy mô hình
    "nhìn vào đâu" khi xử lý từng token — kỹ thuật cơ bản trong
    Circuit Study (Method 6 trong tutorial).
    
    Args:
        attention_weights: Ma trận attention [seq_len, seq_len]
        tokens: Danh sách token đã decode
        layer: Index của layer (0-indexed)
        head: Index của attention head (0-indexed)
        save_path: Đường dẫn lưu file ảnh (nếu None, không lưu)
        
    Returns:
        plt.Figure: Figure object của matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="magma",
        ax=ax,
        square=True,
        cbar_kws={"label": "Attention Weight", "shrink": 0.8},
        linewidths=0.5,
        linecolor="#1a1a2e",
    )
    
    ax.set_title(
        f"Attention Pattern — Layer {layer}, Head {head}",
        fontsize=14,
        fontweight="bold",
        color="#e94560",
        pad=15,
    )
    ax.set_xlabel("Key (source token)", fontsize=11)
    ax.set_ylabel("Query (target token)", fontsize=11)
    
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[✓] Saved attention heatmap to {save_path}")
    
    return fig


def plot_probe_accuracy_per_layer(
    accuracies: List[float],
    f1_scores: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ biểu đồ accuracy/F1 của Linear Probe theo từng layer.
    
    Cho thấy layer nào trong mô hình "hiểu" thuộc tính mục tiêu
    (ví dụ: sentiment) mạnh nhất — tương ứng với Feature Study 
    roadmap trong tutorial (Gurnee+23).
    
    Args:
        accuracies: Accuracy tại mỗi layer [n_layers]
        f1_scores: F1-score tại mỗi layer (tùy chọn)
        save_path: Đường dẫn lưu file ảnh
        
    Returns:
        plt.Figure: Figure object
    """
    n_layers = len(accuracies)
    layers = list(range(n_layers))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bar_width = 0.35
    x = np.arange(n_layers)
    
    bars1 = ax.bar(
        x - bar_width / 2, accuracies, bar_width,
        label="Accuracy", color="#e94560", alpha=0.9,
        edgecolor="#fff", linewidth=0.5,
    )
    
    if f1_scores is not None:
        bars2 = ax.bar(
            x + bar_width / 2, f1_scores, bar_width,
            label="F1-Score", color="#0f3460", alpha=0.9,
            edgecolor="#fff", linewidth=0.5,
        )
    
    # Highlight layer có accuracy cao nhất
    best_layer = int(np.argmax(accuracies))
    ax.axvline(x=best_layer, color="#53d8fb", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.annotate(
        f"Best: Layer {best_layer}\n({accuracies[best_layer]:.1%})",
        xy=(best_layer, accuracies[best_layer]),
        xytext=(best_layer + 1.5, max(accuracies) * 0.95),
        fontsize=10, color="#53d8fb",
        arrowprops=dict(arrowstyle="->", color="#53d8fb", lw=1.5),
    )
    
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Linear Probe Performance per Layer",
        fontsize=14, fontweight="bold", color="#e94560", pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in layers])
    ax.legend(framealpha=0.3, edgecolor="#555")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[✓] Saved probe accuracy chart to {save_path}")
    
    return fig


def plot_sae_loss_curve(
    total_losses: List[float],
    recon_losses: List[float],
    sparsity_losses: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ đường cong hội tụ Loss của SAE trong quá trình huấn luyện.
    
    Hiển thị 3 thành phần: Total Loss, Reconstruction Loss (MSE),
    và Sparsity Loss (L1) — tương ứng với Loss Function cốt lõi
    L(h,h') = ||h-h'||² + λ||z||₁ trong tutorial (Bricken+23).
    
    Args:
        total_losses: Tổng loss mỗi epoch
        recon_losses: Reconstruction loss (MSE) mỗi epoch
        sparsity_losses: Sparsity loss (L1) mỗi epoch
        save_path: Đường dẫn lưu file ảnh
        
    Returns:
        plt.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(total_losses) + 1)
    
    ax.plot(epochs, total_losses, color="#e94560", linewidth=2.5, label="Total Loss", marker="o", markersize=3)
    ax.plot(epochs, recon_losses, color="#53d8fb", linewidth=1.8, label="Reconstruction (MSE)", linestyle="--")
    ax.plot(epochs, sparsity_losses, color="#f5a623", linewidth=1.8, label="Sparsity (L1)", linestyle=":")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(
        "SAE Training Loss Convergence",
        fontsize=14, fontweight="bold", color="#e94560", pad=15,
    )
    ax.legend(framealpha=0.3, edgecolor="#555")
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[✓] Saved SAE loss curve to {save_path}")
    
    return fig


def plot_sparsity_histogram(
    activations: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ histogram phân bố tần suất kích hoạt (sparsity) của các SAE features.
    
    Feature tốt: phần lớn values = 0 (sparse), chỉ một số ít kích hoạt.
    Dead features: features luôn = 0 trên mọi input.
    
    Args:
        activations: Ma trận activation của SAE [n_samples, n_features]
        save_path: Đường dẫn lưu file ảnh
        
    Returns:
        plt.Figure: Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram 1: Phân bố tỷ lệ kích hoạt (mỗi feature kích hoạt bao nhiêu % samples)
    feature_activation_rate = (activations > 0).mean(axis=0)
    
    axes[0].hist(
        feature_activation_rate, bins=50, color="#e94560", alpha=0.85,
        edgecolor="#1a1a2e", linewidth=0.5,
    )
    axes[0].set_xlabel("Activation Rate (% samples)", fontsize=11)
    axes[0].set_ylabel("Number of Features", fontsize=11)
    axes[0].set_title("Feature Activation Rate Distribution", fontsize=12, fontweight="bold", color="#e94560")
    
    dead_features = (feature_activation_rate == 0).sum()
    total_features = len(feature_activation_rate)
    axes[0].annotate(
        f"Dead Features: {dead_features}/{total_features}\n({dead_features/total_features:.1%})",
        xy=(0.95, 0.95), xycoords="axes fraction",
        fontsize=10, color="#f5a623", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#f5a623", alpha=0.8),
    )
    
    # Histogram 2: Phân bố L0 sparsity (mỗi sample kích hoạt bao nhiêu features)
    l0_per_sample = (activations > 0).sum(axis=1)
    
    axes[1].hist(
        l0_per_sample, bins=50, color="#53d8fb", alpha=0.85,
        edgecolor="#1a1a2e", linewidth=0.5,
    )
    axes[1].set_xlabel("Number of Active Features (L0)", fontsize=11)
    axes[1].set_ylabel("Number of Samples", fontsize=11)
    axes[1].set_title("L0 Sparsity Distribution per Sample", fontsize=12, fontweight="bold", color="#53d8fb")
    axes[1].annotate(
        f"Mean L0: {l0_per_sample.mean():.1f}",
        xy=(0.95, 0.95), xycoords="axes fraction",
        fontsize=10, color="#f5a623", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#f5a623", alpha=0.8),
    )
    
    plt.suptitle(
        "SAE Sparsity Analysis",
        fontsize=14, fontweight="bold", color="#e94560", y=1.02,
    )
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[✓] Saved sparsity histogram to {save_path}")
    
    return fig


def plot_logit_lens(
    top_tokens_per_layer: List[List[str]],
    top_probs_per_layer: List[List[float]],
    input_token: str,
    position: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ Logit Lens visualization — Method 2 (Vocabulary Projection) trong tutorial.
    
    Hiển thị top-K predicted tokens tại mỗi layer cho một position cụ thể,
    cho thấy cách mô hình "tinh chỉnh" dự đoán qua từng tầng xử lý.
    
    Tham khảo: nostalgebraist (2020), "Interpreting GPT: the Logit Lens"
    
    Args:
        top_tokens_per_layer: Top tokens dự đoán tại mỗi layer [[tokens_L0], [tokens_L1], ...]
        top_probs_per_layer: Xác suất tương ứng
        input_token: Token đầu vào tại position đang xét
        position: Vị trí token trong chuỗi
        save_path: Đường dẫn lưu file ảnh
        
    Returns:
        plt.Figure: Figure object
    """
    n_layers = len(top_tokens_per_layer)
    k = len(top_tokens_per_layer[0]) if top_tokens_per_layer else 5
    
    fig, ax = plt.subplots(figsize=(14, max(6, n_layers * 0.5)))
    
    # Tạo bảng text annotation
    y_positions = list(range(n_layers))
    
    for layer_idx in range(n_layers):
        tokens = top_tokens_per_layer[layer_idx]
        probs = top_probs_per_layer[layer_idx]
        
        # Vẽ bar cho probability của top-1 token
        ax.barh(layer_idx, probs[0], color="#e94560", alpha=0.7, height=0.6)
        
        # Annotate với top tokens
        token_str = " | ".join([f"'{t}' ({p:.2%})" for t, p in zip(tokens[:3], probs[:3])])
        ax.annotate(
            token_str,
            xy=(probs[0] + 0.01, layer_idx),
            fontsize=9, color="#eee", va="center",
        )
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Layer {i}" for i in range(n_layers)])
    ax.set_xlabel("Top-1 Probability", fontsize=12)
    ax.set_title(
        f'Logit Lens — Prediction Evolution at Position {position} (input: "{input_token}")',
        fontsize=13, fontweight="bold", color="#e94560", pad=15,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[✓] Saved logit lens plot to {save_path}")
    
    return fig


def plot_steering_comparison(
    original_probs: dict,
    steered_probs: dict,
    top_k: int = 15,
    steering_coeff: float = 0.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    So sánh phân bố xác suất từ vựng trước và sau khi steering.
    
    Cho thấy tác động của can thiệp nhân quả (Causal Intervention)
    lên output distribution — evaluation technique trong tutorial.
    
    Args:
        original_probs: Dict {token: probability} của output gốc
        steered_probs: Dict {token: probability} của output sau steering
        top_k: Số lượng token hiển thị
        steering_coeff: Hệ số can thiệp c
        save_path: Đường dẫn lưu file ảnh
        
    Returns:
        plt.Figure: Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sort và lấy top-k
    orig_sorted = sorted(original_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    steer_sorted = sorted(steered_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Original
    tokens_orig, probs_orig = zip(*orig_sorted) if orig_sorted else ([], [])
    axes[0].barh(range(len(tokens_orig)), probs_orig, color="#53d8fb", alpha=0.85)
    axes[0].set_yticks(range(len(tokens_orig)))
    axes[0].set_yticklabels(tokens_orig, fontsize=9)
    axes[0].set_title("Original Output", fontsize=12, fontweight="bold", color="#53d8fb")
    axes[0].set_xlabel("Probability")
    axes[0].invert_yaxis()
    
    # Steered
    tokens_steer, probs_steer = zip(*steer_sorted) if steer_sorted else ([], [])
    axes[1].barh(range(len(tokens_steer)), probs_steer, color="#e94560", alpha=0.85)
    axes[1].set_yticks(range(len(tokens_steer)))
    axes[1].set_yticklabels(tokens_steer, fontsize=9)
    axes[1].set_title(f"Steered Output (c={steering_coeff})", fontsize=12, fontweight="bold", color="#e94560")
    axes[1].set_xlabel("Probability")
    axes[1].invert_yaxis()
    
    plt.suptitle(
        "Next-Token Probability Distribution: Before vs After Steering",
        fontsize=14, fontweight="bold", color="#eee", y=1.02,
    )
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[✓] Saved steering comparison to {save_path}")
    
    return fig
