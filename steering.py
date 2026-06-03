"""
steering.py — Giai đoạn 4: Can thiệp Nhân quả & Bẻ lái mô hình
=================================================================

Thực hiện Causal Intervention (Can thiệp Nhân quả) để chứng minh
mối quan hệ nhân quả giữa Feature Direction Vector và hành vi mô hình.

Công thức can thiệp:
    h_mới = h_cũ + c × v_feature
    
Trong đó:
    - h_cũ: Hidden state gốc tại target layer
    - c: Hệ số can thiệp (steering coefficient)
    - v_feature: Feature Direction Vector (từ Probe hoặc SAE)

Kỹ thuật tham chiếu trong tutorial:
  - Causal-Intervention Evaluation (Part 2.1): "What happens if we 
    manually increase the activation of a feature?"
  - Circuit Study - Method 5: Intervention for Validation
  - Ablation: h_new = h_old - projection(h_old, v_feature)

Yếu tố "vượt xa yêu cầu":
  - Không chỉ quan sát (passive) mà can thiệp chủ động (active)
  - Chứng minh NHÂN QUẢ thay vì chỉ tương quan

Output:
  - outputs/steering_comparison.png: So sánh output trước/sau steering
  - Terminal: Generated text ở các mức c khác nhau

Người phụ trách: Tuấn
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.model_utils import load_model_and_tokenizer, get_device, set_seed
from utils.viz_utils import plot_steering_comparison


class ActivationSteerer:
    """
    Can thiệp vào hidden state của mô hình trong quá trình sinh text.
    
    Sử dụng PyTorch Forward Hook để:
    1. Chặn hidden state tại target layer
    2. Cộng/trừ Feature Direction Vector vào hidden state
    3. Cho mô hình tiếp tục sinh text với hidden state đã bị "bẻ lái"
    
    Attributes:
        model: Mô hình GPT-2
        tokenizer: Tokenizer
        device: Thiết bị tính toán
        target_layer: Layer sẽ bị can thiệp
        steering_vector: Feature Direction Vector
        steering_coeff: Hệ số can thiệp c
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        target_layer: int,
        steering_vector: torch.Tensor,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.target_layer = target_layer
        self.steering_vector = steering_vector.to(device)
        self.steering_coeff = 0.0
        self._hook_handle = None
    
    def _steering_hook(self, module, input, output):
        """
        Hook function: Can thiệp vào hidden state.
        
        Công thức: h_new = h_old + c × v_feature
        
        Áp dụng cho TẤT CẢ token positions trong sequence.
        steering_vector được broadcast lên [1, seq_len, d_model].
        """
        is_tuple = isinstance(output, tuple)
        hidden_state = output[0] if is_tuple else output  # [batch, seq_len, d_model]
        
        # Broadcast steering vector: [d_model] → [1, 1, d_model]
        direction = self.steering_vector.unsqueeze(0).unsqueeze(0)
        
        # Can thiệp: h_new = h_old + c × v
        modified = hidden_state + self.steering_coeff * direction
        
        # Trả về output tuple hoặc Tensor tương ứng
        if is_tuple:
            return (modified,) + output[1:]
        return modified
    
    def _ablation_hook(self, module, input, output):
        """
        Hook function: Ablation — loại bỏ hoàn toàn thành phần feature.
        
        Công thức: h_new = h_old - projection(h_old, v_feature)
        projection(h, v) = (h · v) × v / ||v||²
        
        Nếu sau ablation, mô hình mất khả năng → chứng minh feature 
        thực sự được sử dụng (không chỉ "present" mà còn "used").
        """
        is_tuple = isinstance(output, tuple)
        hidden_state = output[0] if is_tuple else output
        v = self.steering_vector
        
        # Tính projection lên feature direction
        # proj = (h · v) / ||v||² × v
        dot_product = torch.sum(hidden_state * v, dim=-1, keepdim=True)
        v_norm_sq = torch.sum(v * v)
        projection = (dot_product / v_norm_sq) * v
        
        # Loại bỏ projection
        modified = hidden_state - projection
        
        if is_tuple:
            return (modified,) + output[1:]
        return modified
    
    def generate_with_steering(
        self,
        prompt: str,
        steering_coeff: float = 0.0,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True,
        use_ablation: bool = False,
    ) -> str:
        """
        Sinh text với can thiệp tại target layer.
        
        Args:
            prompt: Chuỗi prompt đầu vào
            steering_coeff: Hệ số can thiệp c
                - c > 0: Tăng cường feature (ví dụ: ép positive sentiment)
                - c < 0: Triệt tiêu feature (ví dụ: ép negative sentiment)
                - c = 0: Không can thiệp (output gốc)
            max_new_tokens: Số token tối đa sinh ra
            temperature: Nhiệt độ sampling
            do_sample: Có sampling hay dùng greedy
            use_ablation: Nếu True, dùng ablation thay vì steering
            
        Returns:
            str: Text đã sinh
        """
        self.steering_coeff = steering_coeff
        
        # Đăng ký hook tại target layer
        block = self.model.transformer.h[self.target_layer]
        hook_fn = self._ablation_hook if use_ablation else self._steering_hook
        self._hook_handle = block.register_forward_hook(hook_fn)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        
        finally:
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None
    
    def get_next_token_probs(
        self,
        prompt: str,
        steering_coeff: float = 0.0,
        top_k: int = 20,
    ) -> Dict[str, float]:
        """
        Lấy phân bố xác suất next-token (trước và sau steering).
        
        Dùng để visualize sự thay đổi trong output distribution
        khi áp dụng can thiệp — tương ứng với Causal-Intervention 
        Evaluation trong tutorial.
        
        Args:
            prompt: Chuỗi prompt
            steering_coeff: Hệ số can thiệp
            top_k: Số token top xác suất trả về
            
        Returns:
            Dict[str, float]: {token: probability}
        """
        self.steering_coeff = steering_coeff
        
        block = self.model.transformer.h[self.target_layer]
        self._hook_handle = block.register_forward_hook(self._steering_hook)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits
                probs = F.softmax(logits, dim=-1)
            
            top_probs, top_indices = probs.topk(top_k)
            
            result = {}
            for prob, idx in zip(top_probs, top_indices):
                token = self.tokenizer.decode([idx.item()])
                result[token] = prob.item()
            
            return result
        
        finally:
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None


def main():
    """
    Pipeline chính: Demo can thiệp nhân quả ở các mức c khác nhau.
    
    Bước 1: Tải model + feature direction
    Bước 2: Sinh text ở các mức c = {-10, -5, 0, +5, +10}
    Bước 3: So sánh phân bố xác suất
    Bước 4: Demo ablation
    Bước 5: Visualize
    """
    parser = argparse.ArgumentParser(description="Steering / Causal Intervention")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--prompt", type=str, default="The movie was", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Bước 1: Tải model và feature direction
    model, tokenizer = load_model_and_tokenizer("gpt2", device)
    
    direction_path = os.path.join(args.checkpoint_dir, "feature_direction.pt")
    if not os.path.exists(direction_path):
        print("[!] feature_direction.pt not found. Run linear_probe.py first.")
        return
    
    direction_data = torch.load(direction_path, weights_only=False)
    feature_direction = direction_data["direction"]
    target_layer = direction_data["layer"]
    
    print(f"[*] Feature direction from: {direction_data['source']}")
    print(f"[*] Target layer: {target_layer}")
    print(f"[*] Prompt: \"{args.prompt}\"")
    
    # Tạo steerer
    steerer = ActivationSteerer(model, tokenizer, device, target_layer, feature_direction)
    
    # Bước 2: Sinh text ở các mức can thiệp
    coefficients = [-10, -5, -2, 0, 2, 5, 10]
    
    print(f"\n{'='*70}")
    print("STEERING DEMO: Sentiment Direction")
    print(f"{'='*70}")
    
    for c in coefficients:
        text = steerer.generate_with_steering(
            args.prompt,
            steering_coeff=c,
            max_new_tokens=args.max_tokens,
            temperature=0.7,
        )
        direction_label = "POSITIVE" if c > 0 else "NEGATIVE" if c < 0 else "NEUTRAL"
        print(f"\n[c={c:+.0f}] ({direction_label}):")
        print(f"  {text}")
    
    # Bước 3: So sánh phân bố xác suất
    print(f"\n{'='*70}")
    print("NEXT-TOKEN PROBABILITY COMPARISON")
    print(f"{'='*70}")
    
    original_probs = steerer.get_next_token_probs(args.prompt, steering_coeff=0)
    steered_positive_probs = steerer.get_next_token_probs(args.prompt, steering_coeff=10)
    steered_negative_probs = steerer.get_next_token_probs(args.prompt, steering_coeff=-10)
    
    print("\nOriginal (c=0) top tokens:")
    for token, prob in list(original_probs.items())[:10]:
        print(f"  '{token}': {prob:.4f}")
    
    print(f"\nSteered positive (c=+10) top tokens:")
    for token, prob in list(steered_positive_probs.items())[:10]:
        print(f"  '{token}': {prob:.4f}")
    
    print(f"\nSteered negative (c=-10) top tokens:")
    for token, prob in list(steered_negative_probs.items())[:10]:
        print(f"  '{token}': {prob:.4f}")
    
    # Bước 4: Visualize
    plot_steering_comparison(
        original_probs, steered_positive_probs,
        steering_coeff=10,
        save_path=os.path.join(args.output_dir, "steering_positive.png"),
    )
    
    plot_steering_comparison(
        original_probs, steered_negative_probs,
        steering_coeff=-10,
        save_path=os.path.join(args.output_dir, "steering_negative.png"),
    )
    
    # Bước 5: Demo ablation
    print(f"\n{'='*70}")
    print("ABLATION DEMO: Removing sentiment direction")
    print(f"{'='*70}")
    
    ablated_text = steerer.generate_with_steering(
        args.prompt,
        steering_coeff=0,
        max_new_tokens=args.max_tokens,
        use_ablation=True,
    )
    original_text = steerer.generate_with_steering(
        args.prompt,
        steering_coeff=0,
        max_new_tokens=args.max_tokens,
    )
    
    print(f"\nOriginal: {original_text}")
    print(f"\nAblated:  {ablated_text}")
    print(f"\n[*] If ablation changes the output significantly,")
    print(f"    it proves the feature direction is CAUSALLY USED by the model.")


if __name__ == "__main__":
    main()
