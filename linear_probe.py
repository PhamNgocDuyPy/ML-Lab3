"""
linear_probe.py — Giai đoạn 2: Khảo sát và Định vị không gian đặc trưng
=========================================================================

Sử dụng kỹ thuật Linear Probing (Method 1 trong tutorial) để:
  1. Huấn luyện một Linear Probe (Logistic Regression) từ đầu bằng PyTorch
     cho mỗi layer của GPT-2
  2. Đánh giá Accuracy/F1 trên tập test → Xác định layer nào "hiểu" sentiment
  3. Trích xuất Feature Direction Vector từ layer tốt nhất

Kỹ thuật tham chiếu trong tutorial:
  - Feature Study - Method 1: Probing (Gurnee+23, Hewitt&Liang 2019)
  - "A high probing accuracy on a held-out test set indicates the presence of feature"

Hàm Loss: Binary Cross-Entropy (cho binary classification)
Tham số học: Learning rate, Epochs (không dùng thư viện bọc sẵn)

Output:
  - outputs/probe_accuracy_per_layer.png: Biểu đồ accuracy/F1 theo layer
  - checkpoints/feature_direction.pt: Vector hướng đặc trưng sentiment
  - checkpoints/probe_results.pt: Kết quả chi tiết

Người phụ trách: Kim
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import List, Tuple, Dict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.model_utils import set_seed
from utils.viz_utils import plot_probe_accuracy_per_layer


class LinearProbe(nn.Module):
    """
    Linear Probe (Logistic Regression) tự thiết kế từ đầu bằng PyTorch.
    
    Kiến trúc đơn giản: một lớp Linear duy nhất ánh xạ từ không gian
    activation (d_model chiều) sang 1 output (positive/negative).
    
    Lý do dùng Linear Probe thay vì MLP:
    - Nếu Linear Probe đạt accuracy cao → feature đã được mã hóa LINEAR
      trong activation → mô hình thực sự "hiểu" feature đó.
    - Nếu phải dùng MLP mới đạt accuracy → có thể probe tự học feature,
      không phải feature có sẵn trong activation (Hewitt & Liang, 2019).
    
    Trọng số (weight vector) sau khi train chính là Feature Direction Vector:
      - Hướng trong không gian d_model mà sentiment được mã hóa
      - Dùng cho Steering ở Giai đoạn 4
    
    Args:
        input_dim: Số chiều đầu vào (d_model = 768 cho GPT-2 Small)
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        # Khởi tạo trọng số theo Xavier initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Linear → Sigmoid.
        
        Args:
            x: Activation vector [batch_size, d_model]
            
        Returns:
            Xác suất positive sentiment [batch_size, 1]
        """
        return torch.sigmoid(self.linear(x))
    
    def get_feature_direction(self) -> torch.Tensor:
        """
        Trích xuất Feature Direction Vector (trọng số của linear layer).
        
        Vector này chỉ hướng trong không gian activation mà sentiment
        thay đổi mạnh nhất. Dùng cho Steering: h_new = h_old + c * v_feature
        
        Returns:
            Feature direction vector [d_model] (đã normalize)
        """
        direction = self.linear.weight.data[0].clone()
        # Normalize to unit vector
        direction = direction / direction.norm()
        return direction


def train_probe_for_layer(
    activations: torch.Tensor,
    labels: torch.Tensor,
    layer_idx: int,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Tuple[LinearProbe, float, float]:
    """
    Huấn luyện Linear Probe cho một layer cụ thể.
    
    Quy trình:
      1. Chia train/test (80/20)
      2. Huấn luyện probe bằng BCE Loss + Adam optimizer
      3. Đánh giá trên test set
    
    Args:
        activations: Activation tensor [n_samples, d_model]
        labels: Label tensor [n_samples]
        layer_idx: Index của layer đang train
        lr: Learning rate
        epochs: Số epoch
        batch_size: Batch size
        device: Thiết bị tính toán
        
    Returns:
        Tuple: (trained_probe, test_accuracy, test_f1)
    """
    # Chia train/test
    n = len(activations)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels.numpy())
    
    X_train = activations[train_idx].to(device)
    X_test = activations[test_idx].to(device)
    y_train = labels[train_idx].float().to(device)
    y_test = labels[test_idx].float().to(device)
    
    # Khởi tạo probe
    d_model = activations.shape[1]
    probe = LinearProbe(d_model).to(device)
    
    # Loss function và optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop
    probe.train()
    for epoch in range(epochs):
        # Shuffle indices mỗi epoch
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i + batch_size]
            x_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            optimizer.zero_grad()
            pred = probe(x_batch).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
    
    # Evaluation
    probe.eval()
    with torch.no_grad():
        test_pred = probe(X_test).squeeze(-1)
        test_pred_labels = (test_pred > 0.5).long().cpu().numpy()
        y_test_np = y_test.cpu().numpy().astype(int)
        
        accuracy = accuracy_score(y_test_np, test_pred_labels)
        f1 = f1_score(y_test_np, test_pred_labels, average="binary")
    
    return probe, accuracy, f1


def main():
    """
    Pipeline chính: Train Linear Probe cho mỗi layer → Tìm best layer → Trích xuất Feature Direction.
    
    Bước 1: Tải activations từ Giai đoạn 1
    Bước 2: Train probe cho mỗi layer (12 layers)
    Bước 3: So sánh accuracy → Tìm layer tốt nhất
    Bước 4: Trích xuất Feature Direction Vector
    Bước 5: Visualize và lưu kết quả
    """
    parser = argparse.ArgumentParser(description="Train Linear Probes for each layer")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Bước 1: Tải activations
    activation_path = os.path.join(args.checkpoint_dir, "activations.pt")
    if not os.path.exists(activation_path):
        print(f"[!] File {activation_path} not found. Run extract_hooks.py first.")
        return
    
    data = torch.load(activation_path, weights_only=False)
    all_activations = data["activations"]  # [n_samples, n_layers, d_model]
    labels = data["labels"]  # [n_samples]
    
    n_samples, n_layers, d_model = all_activations.shape
    print(f"[*] Loaded {n_samples} samples, {n_layers} layers, d_model={d_model}")
    
    # Chuẩn hóa activations theo từng layer để tối ưu hóa hội tụ
    mean = all_activations.mean(dim=0, keepdim=True)
    std = all_activations.std(dim=0, keepdim=True) + 1e-8
    all_activations = (all_activations - mean) / std
    
    # Bước 2: Train probe cho mỗi layer
    accuracies = []
    f1_scores = []
    probes = {}
    
    print(f"\n{'='*60}")
    print(f"{'Layer':<8} {'Accuracy':<12} {'F1-Score':<12}")
    print(f"{'='*60}")
    
    for layer_idx in range(n_layers):
        layer_activations = all_activations[:, layer_idx, :]  # [n_samples, d_model]
        
        probe, acc, f1 = train_probe_for_layer(
            layer_activations, labels, layer_idx,
            lr=args.lr, epochs=args.epochs, device=device,
        )
        
        accuracies.append(acc)
        f1_scores.append(f1)
        probes[layer_idx] = probe
        
        marker = " ★" if acc == max(accuracies) else ""
        print(f"Layer {layer_idx:<3} {acc:<12.4f} {f1:<12.4f}{marker}")
    
    print(f"{'='*60}")
    
    # Bước 3: Tìm layer tốt nhất
    best_layer = int(np.argmax(accuracies))
    best_acc = accuracies[best_layer]
    best_f1 = f1_scores[best_layer]
    
    print(f"\n[★] Best layer: {best_layer} (Accuracy: {best_acc:.4f}, F1: {best_f1:.4f})")
    
    # Bước 4: Trích xuất Feature Direction Vector từ best layer
    best_probe = probes[best_layer]
    feature_direction = best_probe.get_feature_direction().cpu()
    
    print(f"[*] Feature Direction Vector shape: {feature_direction.shape}")
    print(f"    L2 norm: {feature_direction.norm():.4f}")
    
    # Bước 5: Visualize
    fig = plot_probe_accuracy_per_layer(
        accuracies, f1_scores,
        save_path=os.path.join(args.output_dir, "probe_accuracy_per_layer.png"),
    )
    
    # Lưu kết quả
    results = {
        "accuracies": accuracies,
        "f1_scores": f1_scores,
        "best_layer": best_layer,
        "best_accuracy": best_acc,
        "best_f1": best_f1,
        "feature_direction": feature_direction,
    }
    
    result_path = os.path.join(args.checkpoint_dir, "probe_results.pt")
    torch.save(results, result_path)
    print(f"\n[✓] Saved probe results to {result_path}")
    
    direction_path = os.path.join(args.checkpoint_dir, "feature_direction.pt")
    torch.save({
        "direction": feature_direction,
        "layer": best_layer,
        "source": "linear_probe",
    }, direction_path)
    print(f"[✓] Saved feature direction to {direction_path}")


if __name__ == "__main__":
    main()
