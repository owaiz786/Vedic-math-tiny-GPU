"""
run_all.py — Master script: run all pipeline steps in order

Steps:
  1. Train MLP on MNIST      (src/train.py)
  2. Quantize to Q3.5        (src/quantize.py)
  3. Export to data_mem.mem  (src/export_mem.py)
  4. Generate kernel files   (kernel/kernel.py)

Then simulate with cocotb:
  make SIM=icarus

Then run on PYNQ-Z2:
  python pynq/pynq_infer.py
"""

import subprocess
import sys
import os

steps = [
    ("Train MLP",         ["python", "train.py"]),
    ("Quantize",          ["python", "quantize.py"]),
    ("Export .mem",       ["python", "export_mem.py"]),
    ("Generate kernel",   ["python", "kernel/kernel.py"]),
]

print("=" * 60)
print("  MNIST → TinyGPU FPGA  —  full pipeline")
print("=" * 60)

for name, cmd in steps:
    print(f"\n[{name}]")
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f"  ERROR: step '{name}' failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"  Done.")

print("\n" + "=" * 60)
print("  All preparation steps complete!")
print()
print("  Next steps:")
print("  1. Simulation:")
print("       cd mnist_fpga && make SIM=icarus")
print()
print("  2. On PYNQ-Z2 board:")
print("       scp -r outputs/ xilinx@<board-ip>:~/mnist_fpga/")
print("       scp pynq/pynq_infer.py xilinx@<board-ip>:~/mnist_fpga/")
print("       ssh xilinx@<board-ip>")
print("       cd ~/mnist_fpga && python pynq_infer.py")
print("=" * 60)
