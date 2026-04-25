# Save as test_svd.py
import numpy as np
from skimage import data as skdata
from skimage.transform import resize

# Load image
img64 = resize(skdata.camera(), (64, 64))
print(f"Original image range: {np.min(img64):.3f} - {np.max(img64):.3f}")

# Simulate your fixed-point arithmetic
FIXED_SCALE = 32
fixed_img = np.clip(np.round(img64 * FIXED_SCALE), -128, 127).astype(int)

# Simulate some operations
test_val = fixed_img[32, 32]
print(f"Sample fixed-point value: {test_val}")
print(f"Converted back: {test_val / FIXED_SCALE:.3f}")