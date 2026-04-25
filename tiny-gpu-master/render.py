# render.py
import matplotlib.pyplot as plt
import numpy as np
import re

# 1. Read the TinyGPU hardware trace
with open("test/logs/mandelbrot.log", "r") as f:
    log_data = f.read()

# 2. Extract the final memory values using Regex
# Looking for "Mem[0]: 15", "Mem[1]: 3", etc.
pixels =[]
matches = re.findall(r'Mem\[\d+\]:\s*(\d+)', log_data)

for match in matches:
    pixels.append(int(match))

# 3. Reshape the 256 linear memory addresses back into a 16x16 image grid
if len(pixels) >= 256:
    image_grid = np.array(pixels[:256]).reshape(16, 16)

    # 4. Render the graphics!
    plt.figure(figsize=(6,6))
    plt.imshow(image_grid, cmap='magma', interpolation='nearest')
    plt.title("Mandelbrot Fractal (Rendered via TinyGPU Verilog Simulation)")
    plt.colorbar(label="Hardware Iteration Count")
    plt.savefig("mandelbrot.png", dpi=300, bbox_inches="tight")
    print("Saved output to mandelbrot.png")
else:
    print("Error: GPU did not finish computing all 256 pixels.")