"""
Step 2: Quantize MLP weights to Q3.5 fixed-point (int8, scale=32)

This matches your vedic_multiplier exactly:
  - 8-bit signed input
  - Q3.5 format: 3 integer bits + 5 fractional bits (scale factor = 2^5 = 32)
  - The >>>5 shift is ALREADY fused into vedic_multiplier via full_product[12:5]
    so no extra shift needed in the kernel

Saves quantized arrays to: outputs/weights_q35.npz
"""

import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train import MLP

os.makedirs("outputs", exist_ok=True)

Q35_SCALE = 32.0   # 2^5
INT8_MIN  = -128
INT8_MAX  =  127


def float_to_q35(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a float32 tensor to int8 Q3.5 fixed-point.
    Range representable: [-4.0, +3.96875] in steps of 0.03125
    """
    arr = tensor.detach().numpy()
    scaled = arr * Q35_SCALE
    clipped = np.clip(np.round(scaled), INT8_MIN, INT8_MAX)
    return clipped.astype(np.int8)


def q35_to_float(arr: np.ndarray) -> np.ndarray:
    """Inverse: int8 Q3.5 → float32 (for accuracy checking)"""
    return arr.astype(np.float32) / Q35_SCALE


def quantize():
    # Load trained weights
    model = MLP()
    model.load_state_dict(torch.load("outputs/mlp_mnist.pth", map_location="cpu"))
    model.eval()

    W1 = float_to_q35(model.fc1.weight)   # (128, 784)
    b1 = float_to_q35(model.fc1.bias)     # (128,)
    W2 = float_to_q35(model.fc2.weight)   # (10, 128)
    b2 = float_to_q35(model.fc2.bias)     # (10,)

    print("Quantized shapes:")
    print(f"  W1: {W1.shape}  b1: {b1.shape}")
    print(f"  W2: {W2.shape}  b2: {b2.shape}")

    # Check quantization error
    W1_err = np.abs(q35_to_float(W1) - model.fc1.weight.detach().numpy()).mean()
    W2_err = np.abs(q35_to_float(W2) - model.fc2.weight.detach().numpy()).mean()
    print(f"\nMean quantization error — W1: {W1_err:.5f}  W2: {W2_err:.5f}")

    np.savez("outputs/weights_q35.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    print("Saved: outputs/weights_q35.npz")

    return W1, b1, W2, b2


if __name__ == "__main__":
    quantize()
