"""
Step 3: Export quantized weights + a test image to TinyGPU data memory format

Data memory layout (one byte per address):
  [0x000 .. 0x00F]  16 bytes   — image pixels (4x4 downsampled, Q3.5)
  [0x010 .. 0x08F]  128 bytes  — W1 weights  (16 inputs x 8 neurons, row-major)
  [0x090 .. 0x097]  8 bytes    — b1 biases
  [0x098 .. 0x0C7]  80 bytes   — W2 weights  (8 neurons x 10 outputs, row-major)
  [0x0C8 .. 0x0D1]  10 bytes   — b2 biases
  [0x0D2 .. 0x0DB]  10 bytes   — hidden activations buffer  (written by kernel)
  [0x0DC .. 0x0E5]  10 bytes   — output scores buffer       (written by kernel)

Total: 230 bytes — fits in 8-bit address space (256 bytes)

NOTE: Uses a 4x4 downsampled image (16 pixels) with a tiny net (16->8->10)
so everything fits in DATA_MEM_ADDR_BITS=8 (256 bytes).
To use the full 784-input net, increase DATA_MEM_ADDR_BITS to 18 in gpu.sv.

Outputs:
  outputs/data_mem.mem   — hex file for $readmemh / cocotb Memory.load()
  outputs/test_image.png — the test image used
  outputs/mem_layout.txt — human-readable memory map
"""

import numpy as np
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

os.makedirs("outputs", exist_ok=True)

# ── Memory map constants (byte addresses) ─────────────────────────────────────
IMG_BASE    = 0x000   # 16 bytes  (4x4 image)
W1_BASE     = 0x010   # 128 bytes (16*8)
B1_BASE     = 0x090   # 8 bytes
W2_BASE     = 0x098   # 80 bytes  (8*10)
B2_BASE     = 0x0C8   # 10 bytes
HIDDEN_BASE = 0x0D2   # 10 bytes  (activation buffer, written by GPU)
OUTPUT_BASE = 0x0DC   # 10 bytes  (output scores, written by GPU)
MEM_SIZE    = 256

Q35_SCALE = 32.0


def float_to_q35_byte(val: float) -> int:
    """float32 → unsigned byte representation of signed int8 Q3.5"""
    scaled = int(np.clip(round(val * Q35_SCALE), -128, 127))
    return scaled & 0xFF   # two's complement as unsigned byte


def make_tiny_weights():
    """
    Create a tiny random network (16->8->10) scaled to Q3.5.
    Replace this with real trained weights for better accuracy.
    """
    rng = np.random.default_rng(42)
    W1 = (rng.normal(0, 0.3, (8, 16)) * Q35_SCALE).clip(-128, 127).astype(np.int8)
    b1 = np.zeros(8, dtype=np.int8)
    W2 = (rng.normal(0, 0.3, (10, 8)) * Q35_SCALE).clip(-128, 127).astype(np.int8)
    b2 = np.zeros(10, dtype=np.int8)
    return W1, b1, W2, b2


def load_mnist_sample(index: int = 0):
    """Load one MNIST sample and downsample to 4x4."""
    dataset = datasets.MNIST("data", train=False, download=True,
                              transform=transforms.ToTensor())
    img_tensor, label = dataset[index]
    img_np = img_tensor.squeeze().numpy()   # (28, 28) float in [0,1]

    # Downsample to 4x4 via PIL
    pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
    pil_small = pil_img.resize((4, 4), Image.LANCZOS)
    pil_img.save("outputs/test_image_28x28.png")
    pil_small.save("outputs/test_image_4x4.png")

    pixels_float = np.array(pil_small).astype(np.float32) / 255.0   # [0,1]
    pixels_q35   = np.array([float_to_q35_byte(p) for p in pixels_float.flatten()])
    return pixels_q35, label


def export():
    pixels, label = load_mnist_sample(index=0)
    print(f"Test image label (ground truth): {label}")

    W1, b1, W2, b2 = make_tiny_weights()

    # Build flat memory
    mem = [0] * MEM_SIZE

    # Image
    for i, p in enumerate(pixels):
        mem[IMG_BASE + i] = int(p)

    # W1 (8 x 16, row-major)
    for i, w in enumerate(W1.flatten()):
        mem[W1_BASE + i] = int(w) & 0xFF

    # b1 (8)
    for i, b in enumerate(b1):
        mem[B1_BASE + i] = int(b) & 0xFF

    # W2 (10 x 8, row-major)
    for i, w in enumerate(W2.flatten()):
        mem[W2_BASE + i] = int(w) & 0xFF

    # b2 (10)
    for i, b in enumerate(b2):
        mem[B2_BASE + i] = int(b) & 0xFF

    # Write .mem file (hex, one byte per line — compatible with $readmemh)
    with open("outputs/data_mem.mem", "w") as f:
        for byte in mem:
            f.write(f"{byte:02x}\n")
    print("Saved: outputs/data_mem.mem")

    # Write human-readable layout
    with open("outputs/mem_layout.txt", "w") as f:
        f.write(f"IMG_BASE    = 0x{IMG_BASE:03X}  ({len(pixels)} bytes)\n")
        f.write(f"W1_BASE     = 0x{W1_BASE:03X}  ({W1.size} bytes)\n")
        f.write(f"B1_BASE     = 0x{B1_BASE:03X}  ({b1.size} bytes)\n")
        f.write(f"W2_BASE     = 0x{W2_BASE:03X}  ({W2.size} bytes)\n")
        f.write(f"B2_BASE     = 0x{B2_BASE:03X}  ({b2.size} bytes)\n")
        f.write(f"HIDDEN_BASE = 0x{HIDDEN_BASE:03X}  (8 bytes, written by GPU)\n")
        f.write(f"OUTPUT_BASE = 0x{OUTPUT_BASE:03X}  (10 bytes, written by GPU)\n")
        f.write(f"Ground truth label: {label}\n")
    print("Saved: outputs/mem_layout.txt")

    # Return base addresses as Python ints for use in kernel.py
    return {
        "IMG_BASE":    IMG_BASE,
        "W1_BASE":     W1_BASE,
        "B1_BASE":     B1_BASE,
        "W2_BASE":     W2_BASE,
        "B2_BASE":     B2_BASE,
        "HIDDEN_BASE": HIDDEN_BASE,
        "OUTPUT_BASE": OUTPUT_BASE,
        "label":       label,
    }


if __name__ == "__main__":
    addrs = export()
    print("\nMemory map:")
    for k, v in addrs.items():
        if isinstance(v, int):
            print(f"  {k:14s} = 0x{v:03X}")
