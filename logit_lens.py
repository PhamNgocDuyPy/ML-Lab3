"""
logit_lens.py — Giai đoạn 5 (Bonus): Vocabulary Projection qua từng layer
===========================================================================

Implement Logit Lens (nostalgebraist, 2020) — kỹ thuật Method 2 trong tutorial:
"Decode the next token predictions encoded in the activation"

Công thức:
    LogitLens(h_i^l) = LayerNorm(h_i^l) × W_U^T
    
Trong đó:
    - h_i^l: Hidden state tại layer l, position i
    - W_U: Unembedding matrix (lm_head weights)
    - LayerNorm: Final layer norm của GPT-2

Ý tưởng: Mặc dù unembedding matrix chỉ được train để project final layer,
ta có thể áp dụng nó cho intermediate layers để xem model "đang nghĩ gì"
tại mỗi giai đoạn xử lý.

Kết quả: Bảng token dự đoán theo từng layer cho thấy cách model
tinh chỉnh dần prediction qua các tầng (Belrose+23).

Output:
  - outputs/logit_lens.png: Visualization 
  - Terminal: Bảng top tokens per layer

Giai đoạn bonus — ít effort, nhiều impact cho demo.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.model_utils import load_model_and_tokenizer, get_device, set_seed
from utils.viz_utils import plot_logit_lens


class LogitLens:
    """
    Logit Lens — Vocabulary Projection cho GPT-2.
    
    "Giải mã" hidden state tại mỗi layer bằng cách nhân với 
    unembedding matrix → xem model dự đoán token gì ở mỗi tầng.
    
    Tham khảo:
    - nostalgebraist (2020): "Interpreting GPT: the Logit Lens"
    - Belrose+23: "Eliciting latent predictions with the Tuned Lens"
    
    Lưu ý (từ tutorial):
    "Unembedding matrix was only trained to project the final layer
    representation. Intermediate representations may represent features
    in a different representation space" → Kết quả ở early layers 
    có thể không đáng tin cậy.
    
    Attributes:
        model: GPT-2 model
        tokenizer: Tokenizer
        device: Device
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Lấy unembedding matrix (lm_head) và final layer norm
        self.unembed = model.lm_head.weight.data  # [vocab_size, d_model]
        self.ln_f = model.transformer.ln_f  # Final LayerNorm
    
    def analyze(
        self,
        text: str,
        position: int = -1,
        top_k: int = 5,
    ) -> Dict:
        """
        Chạy Logit Lens cho tất cả layers tại một position.
        
        Quy trình:
        1. Forward pass thu thập hidden states ở mọi layer
        2. Với mỗi layer: LayerNorm → Unembed → Softmax → Top-K
        3. Trả về top tokens & probabilities per layer
        
        Args:
            text: Input text
            position: Position index để phân tích (-1 = last token)
            top_k: Số token top trả về mỗi layer
            
        Returns:
            Dict chứa kết quả phân tích
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Forward pass với output_hidden_states=True
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )
        
        hidden_states = outputs.hidden_states  # Tuple of (n_layers+1) tensors
        # hidden_states[0] = embedding output
        # hidden_states[1..12] = output of each transformer block
        
        tokens = [self.tokenizer.decode([tid]) for tid in inputs.input_ids[0]]
        seq_len = len(tokens)
        
        if position < 0:
            position = seq_len + position  # Convert negative index
        
        input_token = tokens[position] if position < len(tokens) else "N/A"
        
        results = {
            "input_text": text,
            "tokens": tokens,
            "position": position,
            "input_token": input_token,
            "top_tokens_per_layer": [],
            "top_probs_per_layer": [],
            "all_logits_per_layer": [],
        }
        
        n_layers = self.model.config.n_layer
        
        for layer_idx in range(n_layers + 1):
            h = hidden_states[layer_idx][0, position, :]  # [d_model]
            
            # Apply final LayerNorm (quan trọng cho Logit Lens)
            h_normed = self.ln_f(h.unsqueeze(0)).squeeze(0)
            
            # Project to vocabulary space: logits = h × W_U^T
            logits = h_normed @ self.unembed.T  # [vocab_size]
            probs = F.softmax(logits, dim=-1)
            
            top_probs, top_indices = probs.topk(top_k)
            top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices]
            
            layer_name = f"Embed" if layer_idx == 0 else f"Layer {layer_idx - 1}"
            
            results["top_tokens_per_layer"].append(top_tokens)
            results["top_probs_per_layer"].append(top_probs.cpu().tolist())
            results["all_logits_per_layer"].append(logits.cpu())
        
        return results
    
    def analyze_all_positions(
        self,
        text: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Chạy Logit Lens cho TẤT CẢ positions trong sequence.
        
        Args:
            text: Input text
            top_k: Số token top per layer
            
        Returns:
            List[Dict]: Kết quả cho mỗi position
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        seq_len = inputs.input_ids.shape[1]
        
        all_results = []
        for pos in range(seq_len):
            result = self.analyze(text, position=pos, top_k=top_k)
            all_results.append(result)
        
        return all_results


def print_logit_lens_table(results: Dict) -> None:
    """
    In bảng Logit Lens đẹp trên terminal.
    
    Args:
        results: Dict kết quả từ LogitLens.analyze()
    """
    print(f"\n{'='*80}")
    print(f"LOGIT LENS — Position {results['position']} (input: \"{results['input_token']}\")")
    print(f"Input: \"{results['input_text']}\"")
    print(f"{'='*80}")
    print(f"{'Layer':<10} {'Top-1':<15} {'Prob':<8} {'Top-2':<15} {'Top-3':<15}")
    print(f"{'-'*80}")
    
    for layer_idx, (tokens, probs) in enumerate(
        zip(results["top_tokens_per_layer"], results["top_probs_per_layer"])
    ):
        layer_name = "Embed" if layer_idx == 0 else f"Layer {layer_idx - 1}"
        
        t1, t2, t3 = tokens[0], tokens[1], tokens[2]
        p1, p2, p3 = probs[0], probs[1], probs[2]
        
        # Highlight high-confidence predictions
        marker = "★" if p1 > 0.3 else "·" if p1 > 0.1 else " "
        
        print(
            f"{layer_name:<10} "
            f"'{t1}'".ljust(15) + f" {p1:.4f} {marker} "
            f"'{t2}'".ljust(15) +
            f"'{t3}'".ljust(15)
        )
    
    print(f"{'='*80}")


def main():
    """
    Pipeline: Chạy Logit Lens visualization cho GPT-2.
    
    Bước 1: Tải model
    Bước 2: Chạy Logit Lens cho sample prompt
    Bước 3: In bảng kết quả
    Bước 4: Visualize và lưu
    """
    parser = argparse.ArgumentParser(description="Logit Lens Visualization")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Input prompt")
    parser.add_argument("--position", type=int, default=-1, help="Position to analyze (-1 = last)")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K tokens per layer")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Bước 1: Tải model
    model, tokenizer = load_model_and_tokenizer("gpt2", device)
    
    # Bước 2: Chạy Logit Lens
    lens = LogitLens(model, tokenizer, device)
    
    # Demo với nhiều prompt khác nhau
    prompts = [
        args.prompt,
        "The movie was absolutely",
        "1 + 1 =",
        "Paris is the capital of",
    ]
    
    for prompt in prompts:
        results = lens.analyze(prompt, position=args.position, top_k=args.top_k)
        
        # Bước 3: In bảng
        print_logit_lens_table(results)
        
        # Bước 4: Visualize
        # Bỏ layer Embed (index 0) cho visualization gọn hơn
        fig = plot_logit_lens(
            results["top_tokens_per_layer"][1:],  # Skip embed layer
            results["top_probs_per_layer"][1:],
            input_token=results["input_token"],
            position=results["position"],
            save_path=os.path.join(
                args.output_dir,
                f"logit_lens_{prompt.replace(' ', '_')[:30]}.png"
            ),
        )
    
    print(f"\n[✓] Logit Lens analysis complete. Outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
