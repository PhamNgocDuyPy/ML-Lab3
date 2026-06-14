"""
extract_hooks.py — Giai đoạn 1: Thu thập nguyên liệu cấu trúc
==================================================================

Sử dụng PyTorch Forward Hooks để can thiệp trực tiếp vào cấu trúc mô hình
GPT-2 Small, trích xuất toàn bộ:
  1. Residual Stream Activations tại mỗi layer (12 layers)
  2. Attention Weights (attention patterns) tại mỗi layer/head

Kỹ thuật tham chiếu trong tutorial:
  - Feature Study (Part 2.1): Activation extraction là bước đầu tiên
  - Circuit Study (Method 6): Attention Visualization (Cooney & Nanda, 2023)

Output:
  - checkpoints/activations.pt: Tensor activations [n_samples, n_layers, seq_len, d_model]
  - checkpoints/attention_weights.pt: Tensor attention [n_samples, n_layers, n_heads, seq_len, seq_len]
  - outputs/attention_heatmap_L{layer}_H{head}.png

Người phụ trách: Duy
"""

import os
import sys
import argparse
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.model_utils import load_model_and_tokenizer, get_device, set_seed
from utils.viz_utils import plot_attention_heatmap


class ActivationExtractor:
    """
    Trích xuất Residual Stream Activations và Attention Weights từ GPT-2
    sử dụng PyTorch Forward Hooks.
    
    Forward Hook là một callback function được gắn vào một module trong mạng.
    Mỗi khi module đó được gọi forward(), hook sẽ tự động chạy và cho phép
    chúng ta "chụp lại" input/output activation mà không cần sửa đổi mã 
    nguồn mô hình.
    
    Attributes:
        model: Mô hình GPT-2
        tokenizer: Tokenizer
        device: Thiết bị tính toán
        hooks: Danh sách hook handles (để remove sau khi dùng xong)
        residual_stream: Dict lưu activation tại mỗi layer
        attention_patterns: Dict lưu attention weights tại mỗi layer
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hooks = []
        self.residual_stream = {}
        self.attention_patterns = {}
    
    def _register_hooks(self) -> None:
        """
        Đăng ký forward hooks vào tất cả transformer blocks.
        
        Với mỗi block (layer), chúng ta hook vào:
          - Output của toàn bộ block → Residual Stream activation
          - Attention module → Attention weights
        
        Hook function nhận 3 tham số: (module, input, output)
        Chúng ta chỉ quan tâm output để lưu trữ.
        """
        for layer_idx in range(self.model.config.n_layer):
            # Hook residual stream: output của mỗi transformer block
            block = self.model.transformer.h[layer_idx]
            
            def residual_hook(module, input, output, idx=layer_idx):
                # output[0] là hidden state (residual stream)
                self.residual_stream[idx] = output[0].detach().cpu()
            
            handle = block.register_forward_hook(residual_hook)
            self.hooks.append(handle)
            
            # Hook attention: lấy attention weights
            attn = block.attn
            
            def attn_hook(module, input, output, idx=layer_idx):
                # GPT2Attention trả về (attn_output, attn_weights) hoặc (attn_output, present, attn_weights) tùy phiên bản
                # attn_weights có shape [batch, n_heads, seq_len, seq_len]
                attn_weights = output[1] if len(output) > 1 else None
                if attn_weights is not None:
                    self.attention_patterns[idx] = attn_weights.detach().cpu()
            
            handle = attn.register_forward_hook(attn_hook)
            self.hooks.append(handle)
    
    def _remove_hooks(self) -> None:
        """Gỡ bỏ tất cả hooks sau khi sử dụng xong để tránh memory leak."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def _clear_cache(self) -> None:
        """Xóa cache activation để chuẩn bị cho sample tiếp theo."""
        self.residual_stream = {}
        self.attention_patterns = {}
    
    def extract_single(self, text: str) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Trích xuất activations và attention weights cho một câu đầu vào.
        
        Args:
            text: Chuỗi văn bản đầu vào
            
        Returns:
            Tuple: (residual_stream_dict, attention_patterns_dict)
                - residual_stream_dict: {layer_idx: tensor [1, seq_len, d_model]}
                - attention_patterns_dict: {layer_idx: tensor [1, n_heads, seq_len, seq_len]}
        """
        self._clear_cache()
        self._register_hooks()
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # output_attentions=True để GPT2Attention trả về attention weights
                self.model(**inputs, output_attentions=True)
            
            return dict(self.residual_stream), dict(self.attention_patterns)
        finally:
            self._remove_hooks()
    
    def extract_batch(
        self,
        texts: List[str],
        max_length: int = 64,
    ) -> Tuple[torch.Tensor, List[Dict[int, torch.Tensor]]]:
        """
        Trích xuất activations cho một batch các câu văn bản.
        
        Với mỗi câu, lấy hidden state tại vị trí token cuối cùng
        (last token position) — đây là representation tổng hợp toàn bộ
        thông tin của chuỗi, thường dùng cho classification.
        
        Args:
            texts: Danh sách các câu văn bản
            max_length: Độ dài tối đa (tokens)
            
        Returns:
            Tuple:
                - all_activations: tensor [n_samples, n_layers, d_model]
                  (activation tại last token position của mỗi layer)
                - all_attention: List[Dict] attention patterns cho mỗi sample
        """
        n_layers = self.model.config.n_layer
        d_model = self.model.config.n_embd
        n_samples = len(texts)
        
        # Pre-allocate tensor cho activations
        all_activations = torch.zeros(n_samples, n_layers, d_model)
        all_attention = []
        
        print(f"[*] Extracting activations from {n_samples} samples...")
        
        for i, text in enumerate(tqdm(texts, desc="Extracting")):
            residual, attention = self.extract_single(text)
            
            for layer_idx in range(n_layers):
                if layer_idx in residual:
                    # Sử dụng mean pooling qua toàn bộ sequence để thu được ngữ nghĩa tốt hơn cho sentiment
                    all_activations[i, layer_idx] = residual[layer_idx][0].mean(dim=0)
            
            all_attention.append(attention)
        
        print(f"[✓] Extracted activations shape: {all_activations.shape}")
        return all_activations, all_attention
    
    def get_tokens(self, text: str) -> List[str]:
        """
        Tokenize và decode từng token để hiển thị trên visualization.
        
        Args:
            text: Chuỗi văn bản
            
        Returns:
            List[str]: Danh sách token đã decode
        """
        input_ids = self.tokenizer.encode(text)
        return [self.tokenizer.decode([tid]) for tid in input_ids]


def load_sentiment_dataset(
    max_samples: int = 400,
) -> Tuple[List[str], List[int]]:
    """
    Tải dataset SST-2 (Stanford Sentiment Treebank) cho sentiment analysis.
    
    SST-2 là dataset phân loại cảm xúc câu tiếng Anh (positive/negative),
    thường dùng trong nghiên cứu NLP và phù hợp cho Linear Probing trên GPT-2.
    
    Sử dụng split="validation" để lấy các câu hoàn chỉnh thay vì các mảnh cụm từ
    (phrase fragments) trong train split, giúp cải thiện độ chính xác dò tìm (probing accuracy).
    
    Args:
        max_samples: Số lượng mẫu tối đa mỗi class (SST-2 validation có ~400 samples mỗi class)
        
    Returns:
        Tuple[List[str], List[int]]: (texts, labels)
            - labels: 0 = negative, 1 = positive
    """
    from datasets import load_dataset
    
    print("[*] Loading SST-2 validation dataset...")
    dataset = load_dataset("stanfordnlp/sst2", split="validation")
    
    texts = []
    labels = []
    
    # Lấy cân bằng mỗi class
    pos_count = 0
    neg_count = 0
    
    for item in dataset:
        label = item["label"]
        text = item["sentence"]
        
        if label == 1 and pos_count < max_samples:
            texts.append(text)
            labels.append(1)
            pos_count += 1
        elif label == 0 and neg_count < max_samples:
            texts.append(text)
            labels.append(0)
            neg_count += 1
        
        if pos_count >= max_samples and neg_count >= max_samples:
            break
    
    print(f"[✓] Loaded {len(texts)} samples ({pos_count} pos, {neg_count} neg)")
    return texts, labels


def main():
    """
    Pipeline chính: Trích xuất activations và attention weights.
    
    Bước 1: Tải GPT-2 Small
    Bước 2: Tải SST-2 dataset
    Bước 3: Trích xuất activations cho toàn bộ dataset
    Bước 4: Visualize attention heatmap cho câu mẫu
    Bước 5: Lưu kết quả
    """
    parser = argparse.ArgumentParser(description="Extract activations from GPT-2")
    parser.add_argument("--max_samples", type=int, default=400, help="Max samples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Bước 1: Tải mô hình
    model, tokenizer = load_model_and_tokenizer("gpt2", device)
    extractor = ActivationExtractor(model, tokenizer, device)
    
    # Bước 2: Tải dataset
    texts, labels = load_sentiment_dataset(max_samples=args.max_samples)
    
    # Bước 3: Trích xuất activations cho toàn bộ dataset
    all_activations, all_attention = extractor.extract_batch(texts)
    
    # Bước 4: Visualize attention heatmap cho câu mẫu
    sample_text = "The movie was absolutely wonderful and I loved every moment of it"
    print(f"\n[*] Visualizing attention for: \"{sample_text}\"")
    
    tokens = extractor.get_tokens(sample_text)
    residual, attention = extractor.extract_single(sample_text)
    
    # Vẽ attention heatmap cho layer 0 head 7 (previous token head, cf. tutorial Method 6)
    # và layer cuối head 9 (name mover head pattern)
    for layer, head in [(0, 7), (5, 0), (11, 9)]:
        if layer in attention:
            attn_matrix = attention[layer][0, head].numpy()
            plot_attention_heatmap(
                attn_matrix, tokens, layer=layer, head=head,
                save_path=os.path.join(args.output_dir, f"attention_L{layer}_H{head}.png"),
            )
    
    # Bước 5: Lưu kết quả
    save_data = {
        "activations": all_activations,
        "labels": torch.tensor(labels),
        "texts": texts,
    }
    
    activation_path = os.path.join(args.checkpoint_dir, "activations.pt")
    torch.save(save_data, activation_path)
    print(f"\n[✓] Saved activations to {activation_path}")
    print(f"    Shape: {all_activations.shape}")
    print(f"    Labels: {len(labels)} ({sum(labels)} pos, {len(labels) - sum(labels)} neg)")
    
    # Lưu attention cho sample text (dùng cho UI)
    sample_attention_path = os.path.join(args.checkpoint_dir, "sample_attention.pt")
    torch.save({
        "attention": attention,
        "tokens": tokens,
        "text": sample_text,
        "residual": residual,
    }, sample_attention_path)
    print(f"[✓] Saved sample attention to {sample_attention_path}")


if __name__ == "__main__":
    main()
