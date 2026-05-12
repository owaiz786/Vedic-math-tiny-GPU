"""
PYNQ-Z2 inference script
Run this ON the PYNQ-Z2 board (copy this file + outputs/ folder to the board)

Prerequisites on the board:
  pip install pynq numpy pillow

Usage:
  python pynq_infer.py --image outputs/test_image_4x4.png
"""

import argparse
import numpy as np
from PIL import Image

try:
    from pynq import Overlay, MMIO
    ON_BOARD = True
except ImportError:
    print("[WARNING] pynq not found — running in dry-run mode (no hardware)")
    ON_BOARD = False

# ── AXI address offsets (match your Vivado block design) ───────────────────
# Adjust these to your actual AXI-Lite address map from Vivado
PROG_MEM_BASE   = 0x4000_0000
DATA_MEM_BASE   = 0x4001_0000
DCR_OFFSET      = 0x00   # device control register
CTRL_OFFSET     = 0x04   # write 1 to start
STATUS_OFFSET   = 0x08   # reads 1 when done
RESULT_BASE     = 0x0DC  # offset into data memory where output scores start

Q35_SCALE = 32.0
INPUT_DIM = 16   # 4x4 image


def preprocess_image(path: str) -> list:
    """Load any image, resize to 4x4, return list of 16 Q3.5 bytes."""
    img = Image.open(path).convert("L").resize((4, 4), Image.LANCZOS)
    pixels = np.array(img).astype(np.float32) / 255.0
    q35 = [int(np.clip(round(p * Q35_SCALE), -128, 127)) & 0xFF
           for p in pixels.flatten()]
    return q35


def load_mem_file(path: str) -> list:
    """Read outputs/data_mem.mem hex file."""
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


def load_kernel(path: str) -> list:
    """Read a kernel_*.py file and return the program list."""
    ns = {}
    with open(path) as f:
        exec(f.read(), ns)
    # Find the list variable (program_layer1 or program_layer2)
    for k, v in ns.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], int):
            return v
    raise ValueError(f"No program list found in {path}")


def run_inference(image_path: str, bitstream: str = "tiny_gpu.bit"):
    pixels = preprocess_image(image_path)
    print(f"Preprocessed image: {len(pixels)} pixels")

    data_mem = load_mem_file("outputs/data_mem.mem")

    # Patch image pixels into data memory at IMG_BASE=0x000
    for i, p in enumerate(pixels):
        data_mem[0x000 + i] = p

    if not ON_BOARD:
        print("[DRY RUN] Would load bitstream:", bitstream)
        print("[DRY RUN] Would write", len(data_mem), "bytes to data memory")
        print("[DRY RUN] Would write kernel to program memory")
        print("[DRY RUN] Predicted digit: N/A (no hardware)")
        return

    # ── Load bitstream ───────────────────────────────────────────────────────
    ol = Overlay(bitstream)
    gpu = ol.tiny_gpu_0   # adjust to your IP block name in Vivado

    # ── Layer 1: img -> hidden (8 threads) ──────────────────────────────────
    kernel1 = load_kernel("outputs/kernel_layer1.py")

    for i, instr in enumerate(kernel1):
        gpu.write(PROG_MEM_BASE + i * 4, instr)
    for i, byte in enumerate(data_mem):
        gpu.write(DATA_MEM_BASE + i * 4, byte)

    gpu.write(DCR_OFFSET, 8)    # 8 hidden-neuron threads
    gpu.write(CTRL_OFFSET, 1)   # start
    while gpu.read(STATUS_OFFSET) != 1:
        pass
    print("Layer 1 done")

    # ── Layer 2: hidden -> scores (10 threads) ──────────────────────────────
    kernel2 = load_kernel("outputs/kernel_layer2.py")

    for i, instr in enumerate(kernel2):
        gpu.write(PROG_MEM_BASE + i * 4, instr)

    gpu.write(DCR_OFFSET, 10)   # 10 output-neuron threads
    gpu.write(CTRL_OFFSET, 1)   # start
    while gpu.read(STATUS_OFFSET) != 1:
        pass
    print("Layer 2 done")

    # ── Read output scores ───────────────────────────────────────────────────
    scores = []
    for i in range(10):
        raw = gpu.read(DATA_MEM_BASE + (RESULT_BASE + i) * 4)
        signed = raw if raw < 128 else raw - 256
        scores.append(signed)

    predicted = np.argmax(scores)
    print(f"\nOutput scores: {scores}")
    print(f"Predicted digit: {predicted}")
    return predicted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     default="outputs/test_image_4x4.png")
    parser.add_argument("--bitstream", default="tiny_gpu.bit")
    args = parser.parse_args()
    run_inference(args.image, args.bitstream)
