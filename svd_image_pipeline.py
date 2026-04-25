import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import resize

# 1. LOAD A CLASSIC STANDARD IMAGE
# We use the famous 'Cameraman' image, resized to 256x256 for processing
print("Loading standard High-Res image...")
# data.camera() already returns a grayscale image, so no need for rgb2gray
img = data.camera()
img = resize(img, (256, 256))

# 2. DEFINE HARDWARE CONSTRAINTS
BLOCK_SIZE = 16
K_RANK = 3           # SVD Compression level (Keep top 3 features)
FIXED_SCALE = 32.0   # Q3.5 Fixed-Point Format (Hardware constraint)

# Prepare an empty canvas for the GPU to write the decompressed image into
reconstructed_img = np.zeros((256, 256))

print(f"Chopping image into {BLOCK_SIZE}x{BLOCK_SIZE} blocks for GPU Streaming...")

# 3. BLOCK-BY-BLOCK PROCESSING (TILING)
for i in range(0, 256, BLOCK_SIZE):
    for j in range(0, 256, BLOCK_SIZE):
        
        # A. Extract the 16x16 block
        block = img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
        
        # B. Host CPU performs SVD Compression
        U, S, Vt = np.linalg.svd(block, full_matrices=False)
        
        # C. Compress: Keep only the top 'K_RANK' components
        U_k = U[:, :K_RANK]
        S_k = np.diag(S[:K_RANK])
        Vt_k = Vt[:K_RANK, :]
        
        # D. Quantize for the 8-bit GPU (Convert Floats to Q3.5 Fixed-Point)
        U_hw  = np.clip(np.round(U_k * FIXED_SCALE), -128, 127).astype(int)
        S_hw  = np.clip(np.round(S_k * FIXED_SCALE), -128, 127).astype(int)
        Vt_hw = np.clip(np.round(Vt_k * FIXED_SCALE), -128, 127).astype(int)
        
        # =================================================================
        # ⚠️ HARDWARE ABSTRACTION LAYER (HAL)
        # In the real Cocotb testbench, you stream U_hw, S_hw, and Vt_hw 
        # into the GPU's memory here, and trigger the MAC instructions.
        # For this software script, we simulate the GPU's fixed-point math:
        # =================================================================
        
        # GPU Math: U * S
        US_hw = np.dot(U_hw, S_hw) >> 5  # Hardware Arithmetic Shift Right by 5!
        
        # GPU Math: (U*S) * Vt
        reconstructed_block_hw = np.dot(US_hw, Vt_hw) >> 5
        
        # =================================================================
        # END OF HARDWARE EXECUTION
        # =================================================================
        
        # E. Convert back to Floats and stitch into the final image canvas
        reconstructed_img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = reconstructed_block_hw / FIXED_SCALE

# 4. RENDER THE RESULTS SIDE-BY-SIDE
print("Rendering Original vs. GPU-Decompressed Image...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(img, cmap='gray')
ax1.set_title("Original 256x256 Image")
ax1.axis('off')

ax2.imshow(reconstructed_img, cmap='gray')
ax2.set_title(f"GPU Decompressed (SVD Rank {K_RANK})\n8-bit Fixed-Point Q3.5 Math")
ax2.axis('off')

plt.tight_layout()
plt.show()
