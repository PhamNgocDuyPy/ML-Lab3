"""
train_sae.py — Giai đoạn 3: Phân rã triệt tiêu hiện tượng đa nghĩa
=====================================================================

Xây dựng và huấn luyện Sparse Autoencoder (SAE) từ đầu bằng PyTorch
để phân tách các vector activation thành các Đặc trưng Đơn nghĩa
(Monosemantic Features).

Kỹ thuật tham chiếu trong tutorial:
  - Feature Study - Method 4: Sparse Autoencoder (Bricken+23)
  - Superposition Problem: Quá nhiều features, không đủ neurons
    → mô hình dùng tổ hợp tuyến tính để mã hóa (Elhage+22)
  - SAE giải quyết bằng cách project lên không gian chiều cao + ép thưa

Kiến trúc SAE:
  - Encoder: Linear(d_model → expansion × d_model) + ReLU
  - Decoder: Linear(expansion × d_model → d_model)

Hàm Loss (Điểm nhấn toán học):
  L(h, h') = ||h - h'||²₂ + λ||z||₁
  - MSE Loss: Đảm bảo tái cấu trúc chính xác
  - L1 Penalty: Ép hidden representation z đạt độ thưa (sparsity)

Sau khi train:
  - Feature Analysis Dashboard: Top-K activating examples per feature
  - Dead Feature Analysis: Đếm features không bao giờ kích hoạt
  - Sparsity Histogram: Phân bố tần suất kích hoạt

Output:
  - checkpoints/sae_weights.pt: Trọng số SAE
  - outputs/sae_loss_curve.png: Đường cong hội tụ
  - outputs/sae_sparsity_histogram.png: Histogram độ thưa
  - checkpoints/sae_feature_analysis.pt: Kết quả phân tích features

Người phụ trách: Tín
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.model_utils import set_seed
from utils.viz_utils import plot_sae_loss_curve, plot_sparsity_histogram


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) tự xây dựng từ đầu bằng PyTorch.
    
    Kiến trúc:
      Encoder: h ∈ ℝ^d → z ∈ ℝ^s (s >> d)
        z = ReLU(h · W_enc + b_enc)
      Decoder: z ∈ ℝ^s → h' ∈ ℝ^d
        h' = z · W_dec + b_dec
    
    Ý tưởng cốt lõi (Bricken+23):
      - Map activation vào không gian chiều CAO hơn nhưng THƯA hơn
      - Mỗi neuron trong z (ideally) mã hóa đúng MỘT feature
      - Decoder columns (W_dec[j]) = feature directions trong model space
    
    Args:
        d_model: Số chiều activation gốc (768 cho GPT-2 Small)
        expansion_factor: Hệ số mở rộng không gian ẩn (mặc định: 8)
            → hidden_dim = d_model × expansion_factor = 768 × 8 = 6144
        l1_coeff: Hệ số λ cho L1 sparsity penalty (mặc định: 1e-3)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        expansion_factor: int = 8,
        l1_coeff: float = 1e-3,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.hidden_dim = d_model * expansion_factor
        self.l1_coeff = l1_coeff
        
        # Encoder: Linear + ReLU (đưa lên không gian chiều cao thưa)
        self.encoder = nn.Linear(d_model, self.hidden_dim)
        
        # Decoder: Linear (tái cấu trúc về không gian gốc)
        self.decoder = nn.Linear(self.hidden_dim, d_model)
        
        # Khởi tạo trọng số
        # Dùng Kaiming init cho encoder (vì có ReLU) và Xavier cho decoder
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        
        # Normalize decoder weights (mỗi column là unit vector)
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )
    
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Encoder: Map activation vào không gian thưa chiều cao.
        
        Args:
            h: Activation vector [batch_size, d_model]
            
        Returns:
            z: Sparse hidden representation [batch_size, hidden_dim]
        """
        return torch.relu(self.encoder(h))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decoder: Tái cấu trúc activation từ representation thưa.
        
        Args:
            z: Sparse representation [batch_size, hidden_dim]
            
        Returns:
            h_reconstructed: Reconstructed activation [batch_size, d_model]
        """
        return self.decoder(z)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Encode → Decode.
        
        Args:
            h: Input activation [batch_size, d_model]
            
        Returns:
            Tuple: (h_reconstructed, z)
                - h_reconstructed: [batch_size, d_model]
                - z: sparse hidden [batch_size, hidden_dim]
        """
        z = self.encode(h)
        h_reconstructed = self.decode(z)
        return h_reconstructed, z
    
    def compute_loss(
        self, h: torch.Tensor, h_reconstructed: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tính hàm Loss tích hợp: L = MSE + λ·L1
        
        Đây là điểm nhấn toán học của SAE:
          - MSE Loss = ||h - h'||²₂ : Đảm bảo tái cấu trúc faithful
          - L1 Loss = λ·||z||₁ : Ép z thưa (sparse) để monosemantic
        
        Trade-off (Issue 1 trong tutorial): 
          Tăng λ → thưa hơn nhưng reconstruction kém hơn
          Giảm λ → reconstruction tốt hơn nhưng ít thưa
        
        Args:
            h: Input gốc [batch_size, d_model]
            h_reconstructed: Output tái cấu trúc [batch_size, d_model]
            z: Hidden representation thưa [batch_size, hidden_dim]
            
        Returns:
            Tuple: (total_loss, reconstruction_loss, sparsity_loss)
        """
        # MSE Reconstruction Loss
        reconstruction_loss = nn.functional.mse_loss(h_reconstructed, h)
        
        # L1 Sparsity Loss
        sparsity_loss = self.l1_coeff * z.abs().mean()
        
        # Total Loss
        total_loss = reconstruction_loss + sparsity_loss
        
        return total_loss, reconstruction_loss, sparsity_loss
    
    def get_feature_directions(self) -> torch.Tensor:
        """
        Trích xuất Feature Direction Vectors từ decoder weights.
        
        Mỗi cột (column) của W_dec đại diện cho hướng của một feature
        trong không gian model activation. Feature j được biểu diễn
        bởi decoder.weight[:, j] (hoặc decoder.weight[j] nếu transposed).
        
        Returns:
            Feature directions [hidden_dim, d_model]
        """
        return self.decoder.weight.data.clone()


def train_sae(
    activations: torch.Tensor,
    d_model: int = 768,
    expansion_factor: int = 8,
    l1_coeff: float = 1e-3,
    lr: float = 3e-4,
    epochs: int = 50,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> Tuple[SparseAutoencoder, Dict]:
    """
    Huấn luyện Sparse Autoencoder trên activations.
    
    Args:
        activations: Tensor activations [n_samples, d_model]
        d_model: Số chiều model
        expansion_factor: Hệ số mở rộng SAE
        l1_coeff: Hệ số L1 penalty
        lr: Learning rate (dùng Adam)
        epochs: Số epoch
        batch_size: Batch size
        device: Thiết bị tính toán
        
    Returns:
        Tuple: (trained_sae, training_history)
    """
    sae = SparseAutoencoder(d_model, expansion_factor, l1_coeff).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=lr)
    
    activations = activations.to(device)
    n_samples = len(activations)
    
    history = {
        "total_loss": [],
        "recon_loss": [],
        "sparsity_loss": [],
        "mean_l0": [],  # Average number of active features per sample
    }
    
    print(f"[*] Training SAE: d_model={d_model}, hidden_dim={sae.hidden_dim}, λ={l1_coeff}")
    print(f"    Samples: {n_samples}, Epochs: {epochs}, Batch size: {batch_size}")
    
    for epoch in range(epochs):
        sae.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_sparse = 0.0
        epoch_l0 = 0.0
        n_batches = 0
        
        # Shuffle indices
        perm = torch.randperm(n_samples, device=device)
        
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i + batch_size]
            h = activations[batch_idx]
            
            optimizer.zero_grad()
            h_recon, z = sae(h)
            total_loss, recon_loss, sparse_loss = sae.compute_loss(h, h_recon, z)
            total_loss.backward()
            optimizer.step()
            
            # Normalize decoder weights sau mỗi step (quan trọng cho SAE)
            with torch.no_grad():
                sae.decoder.weight.data = nn.functional.normalize(
                    sae.decoder.weight.data, dim=0
                )
            
            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_sparse += sparse_loss.item()
            epoch_l0 += (z > 0).float().sum(dim=1).mean().item()
            n_batches += 1
        
        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        avg_sparse = epoch_sparse / n_batches
        avg_l0 = epoch_l0 / n_batches
        
        history["total_loss"].append(avg_total)
        history["recon_loss"].append(avg_recon)
        history["sparsity_loss"].append(avg_sparse)
        history["mean_l0"].append(avg_l0)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:>3}/{epochs} | "
                f"Total: {avg_total:.6f} | "
                f"Recon: {avg_recon:.6f} | "
                f"Sparse: {avg_sparse:.6f} | "
                f"L0: {avg_l0:.1f}"
            )
    
    return sae, history


def analyze_features(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    texts: List[str],
    tokenizer,
    top_k: int = 5,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Phân tích các learned features sau khi train SAE.
    
    Bước quan trọng trong Feature Study workflow (tutorial Step 2-3):
    Với mỗi feature, tìm các examples kích hoạt mạnh nhất để đặt tên
    và hiểu ý nghĩa của feature đó (giống cách Anthropic/Bricken+23 làm).
    
    Phân tích bao gồm:
      1. Top-K activating examples per feature
      2. Dead features (features không bao giờ kích hoạt)
      3. Feature activation statistics
    
    Args:
        sae: SAE đã train
        activations: Activations gốc [n_samples, d_model]
        texts: Texts tương ứng
        tokenizer: Tokenizer (để decode)
        top_k: Số examples hàng đầu mỗi feature
        device: Thiết bị tính toán
        
    Returns:
        Dict chứa kết quả phân tích
    """
    sae.eval()
    activations = activations.to(device)
    
    print("[*] Analyzing SAE features...")
    
    with torch.no_grad():
        _, z = sae(activations)  # [n_samples, hidden_dim]
        z = z.cpu()
    
    z_numpy = z.numpy()
    n_samples, hidden_dim = z_numpy.shape
    
    # Dead features: features luôn = 0 trên mọi input
    feature_activation_rate = (z_numpy > 0).mean(axis=0)
    dead_mask = feature_activation_rate == 0
    n_dead = dead_mask.sum()
    n_alive = hidden_dim - n_dead
    
    print(f"  Dead features:  {n_dead}/{hidden_dim} ({n_dead/hidden_dim:.1%})")
    print(f"  Alive features: {n_alive}/{hidden_dim} ({n_alive/hidden_dim:.1%})")
    
    # Mean L0 sparsity
    l0_per_sample = (z_numpy > 0).sum(axis=1)
    mean_l0 = l0_per_sample.mean()
    print(f"  Mean L0 sparsity: {mean_l0:.1f} active features per sample")
    
    # Top-K activating examples cho top 20 features (most activated)
    mean_activation = z_numpy.mean(axis=0)
    top_feature_indices = np.argsort(-mean_activation)[:20]  # Top 20 most active features
    
    feature_examples = {}
    for feat_idx in top_feature_indices:
        feat_activations = z_numpy[:, feat_idx]
        top_sample_indices = np.argsort(-feat_activations)[:top_k]
        
        examples = []
        for sample_idx in top_sample_indices:
            if feat_activations[sample_idx] > 0:
                examples.append({
                    "text": texts[sample_idx][:200],  # Truncate
                    "activation": float(feat_activations[sample_idx]),
                })
        
        feature_examples[int(feat_idx)] = examples
    
    analysis = {
        "n_features": hidden_dim,
        "n_dead": int(n_dead),
        "n_alive": int(n_alive),
        "dead_rate": float(n_dead / hidden_dim),
        "mean_l0": float(mean_l0),
        "feature_activation_rate": feature_activation_rate,
        "l0_per_sample": l0_per_sample,
        "feature_examples": feature_examples,
        "top_feature_indices": top_feature_indices.tolist(),
        "z_numpy": z_numpy,
    }
    
    return analysis


def main():
    """
    Pipeline chính: Train SAE → Phân tích features → Visualize.
    
    Bước 1: Tải activations từ best layer (Giai đoạn 2)
    Bước 2: Huấn luyện SAE
    Bước 3: Phân tích features
    Bước 4: Visualize (loss curve + sparsity histogram)
    Bước 5: Lưu kết quả
    """
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--expansion_factor", type=int, default=8, help="SAE expansion factor")
    parser.add_argument("--l1_coeff", type=float, default=1e-3, help="L1 sparsity coefficient")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Bước 1: Tải activations
    activation_path = os.path.join(args.checkpoint_dir, "activations.pt")
    probe_path = os.path.join(args.checkpoint_dir, "probe_results.pt")
    
    if not os.path.exists(activation_path):
        print("[!] activations.pt not found. Run extract_hooks.py first.")
        return
    
    data = torch.load(activation_path, weights_only=False)
    all_activations = data["activations"]
    texts = data["texts"]
    
    # Xác định layer tốt nhất từ probe results
    if os.path.exists(probe_path):
        probe_results = torch.load(probe_path, weights_only=False)
        best_layer = probe_results["best_layer"]
        print(f"[*] Using best layer from probing: Layer {best_layer}")
    else:
        best_layer = 8  # Default: layer gần cuối thường tốt nhất
        print(f"[*] probe_results.pt not found. Using default layer: {best_layer}")
    
    # Lấy activations của best layer
    layer_activations = all_activations[:, best_layer, :]  # [n_samples, d_model]
    d_model = layer_activations.shape[1]
    
    print(f"[*] Activations shape: {layer_activations.shape}")
    
    # Bước 2: Huấn luyện SAE
    sae, history = train_sae(
        layer_activations,
        d_model=d_model,
        expansion_factor=args.expansion_factor,
        l1_coeff=args.l1_coeff,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )
    
    # Bước 3: Phân tích features
    analysis = analyze_features(sae, layer_activations, texts, tokenizer=None, device=device)
    
    # Bước 4: Visualize
    # Loss curve
    plot_sae_loss_curve(
        history["total_loss"],
        history["recon_loss"],
        history["sparsity_loss"],
        save_path=os.path.join(args.output_dir, "sae_loss_curve.png"),
    )
    
    # Sparsity histogram
    plot_sparsity_histogram(
        analysis["z_numpy"],
        save_path=os.path.join(args.output_dir, "sae_sparsity_histogram.png"),
    )
    
    # Bước 5: Lưu kết quả
    # SAE weights
    sae_path = os.path.join(args.checkpoint_dir, "sae_weights.pt")
    torch.save({
        "state_dict": sae.state_dict(),
        "d_model": d_model,
        "expansion_factor": args.expansion_factor,
        "l1_coeff": args.l1_coeff,
        "best_layer": best_layer,
        "history": history,
    }, sae_path)
    print(f"\n[✓] Saved SAE weights to {sae_path}")
    
    # Feature analysis (bỏ z_numpy vì quá lớn)
    analysis_save = {k: v for k, v in analysis.items() if k != "z_numpy"}
    analysis_path = os.path.join(args.checkpoint_dir, "sae_feature_analysis.pt")
    torch.save(analysis_save, analysis_path)
    print(f"[✓] Saved feature analysis to {analysis_path}")
    
    # Print top features
    print(f"\n{'='*60}")
    print("TOP 10 MOST ACTIVATED FEATURES:")
    print(f"{'='*60}")
    for i, feat_idx in enumerate(analysis["top_feature_indices"][:10]):
        examples = analysis["feature_examples"].get(feat_idx, [])
        print(f"\nFeature #{feat_idx}:")
        for ex in examples[:3]:
            print(f"  [{ex['activation']:.3f}] \"{ex['text'][:80]}...\"")


if __name__ == "__main__":
    main()
