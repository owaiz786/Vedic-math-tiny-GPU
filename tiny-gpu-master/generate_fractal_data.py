# generate_fractal_data.py
import numpy as np

# We want a 16x16 grid for our GPU
width, height = 16, 16
fixed_point_scale = 32  # 1.0 in our Q3.5 fixed-point format

# Mandelbrot complex plane bounds
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5

memory_dump =[]

for y in range(height):
    for x in range(width):
        # Map pixel to the complex plane
        cx = x_min + (x / width) * (x_max - x_min)
        cy = y_min + (y / height) * (y_max - y_min)
        
        # Convert to our custom 8-bit GPU hardware format
        cx_hw = int(cx * fixed_point_scale) & 0xFF
        cy_hw = int(cy * fixed_point_scale) & 0xFF
        
        memory_dump.append((cx_hw, cy_hw))

print(f"Generated {len(memory_dump)} pixels for GPU Data Memory!")
# You will copy these numbers into the Cocotb testbench to load the GPU memory.