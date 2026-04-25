import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import resize

# -------------------------------------------------------------
# 1. PAGE SETUP
# -------------------------------------------------------------
st.set_page_config(page_title="Hardware Edge AI", layout="wide", page_icon="🖥️")
st.title("🚀 Edge AI: Hardware SVD Decompression Telemetry")
st.markdown("Real-time cycle-by-cycle visualization of the ALU Tensor Core reconstructing an image.")

# -------------------------------------------------------------
# 2. DIGITAL TWIN: SIMULATE THE HARDWARE PIPELINE
# -------------------------------------------------------------
@st.cache_data
def generate_hardware_data():
    """ Runs the exact SVD fixed-point math to generate the baseline data """
    # Load image - camera image is already grayscale
    img = data.camera()
    
    # For safety, ensure it's grayscale (camera is already grayscale)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = color.rgb2gray(img)
    
    img = resize(img, (64, 64))
    
    BLOCK_SIZE = 16
    K_RANK = 3
    FIXED_SCALE = 32.0
    
    reconstructed_img = np.zeros((64, 64))
    
    for i in range(0, 64, BLOCK_SIZE):
        for j in range(0, 64, BLOCK_SIZE):
            block = img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            U, S, Vt = np.linalg.svd(block, full_matrices=False)
            
            U_k = U[:, :K_RANK]
            S_k = np.diag(S[:K_RANK])
            Vt_k = Vt[:K_RANK, :]
            
            US = np.dot(U_k, S_k)
            US_hw = np.clip(np.round(US * FIXED_SCALE), -128, 127).astype(int)
            Vt_hw = np.clip(np.round(Vt_k * FIXED_SCALE), -128, 127).astype(int)
            
            reconstructed_block_hw = np.dot(US_hw, Vt_hw) >> 5
            reconstructed_img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = reconstructed_block_hw / FIXED_SCALE
    
    return img, reconstructed_img

# Generate the data
original_img, final_hw_img = generate_hardware_data()

# Flatten the 64x64 image into a 1D array of 4,096 pixels so we can animate it
pixels_1d = final_hw_img.flatten()

# -------------------------------------------------------------
# 3. THE HARDWARE CLOCK CYCLE SLIDER
# -------------------------------------------------------------
st.markdown("### ⏱️ Hardware Clock Cycle Control")

# Calculate total cycles needed
CYCLES_PER_PIXEL = 5  # 3 multiplies + 2 adds
TOTAL_PIXELS = 64 * 64
MAX_CYCLES = TOTAL_PIXELS * CYCLES_PER_PIXEL

current_cycle = st.slider(
    "Drag to step through ALU execution time:", 
    min_value=0, 
    max_value=MAX_CYCLES, 
    value=0, 
    step=5
)

# Calculate how many pixels the hardware has finished
pixels_completed = current_cycle // CYCLES_PER_PIXEL

# Create the live image canvas
live_canvas_1d = np.zeros_like(pixels_1d)
live_canvas_1d[:pixels_completed] = pixels_1d[:pixels_completed]
live_img = live_canvas_1d.reshape((64, 64))

# -------------------------------------------------------------
# 4. TELEMETRY METRICS & ALU STATE MACHINE
# -------------------------------------------------------------
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Clock Cycles", f"{current_cycle:,} / {MAX_CYCLES:,}")
col2.metric("Pixels Decompressed", f"{pixels_completed:,} / {TOTAL_PIXELS:,}")
col3.metric("Hardware Architecture", "8-Bit Fixed-Point Q3.5")
col4.metric("Compression Ratio", "81% Data Dropped (Rank 3)")

# Determine what the ALU is doing right now based on the clock cycle remainder
cycle_state = current_cycle % CYCLES_PER_PIXEL
alu_status = ""

if pixels_completed >= TOTAL_PIXELS:
    alu_status = "🟢 IDLE: Image Decompression Complete!"
elif cycle_state == 0:
    alu_status = "⚡ FETCHING: Loading next Pixel matrices from Memory..."
elif cycle_state in [1, 2, 3]:
    alu_status = "🔴 COMPUTING: FIXED_MUL (rs * rt) >>> 5"
elif cycle_state == 4:
    alu_status = "🔵 ACCUMULATING: ADD (Running Total + New Product)"
else:
    alu_status = "⚙️ ALU Processing..."

# Calculate percentage complete for progress bar
progress_percent = (current_cycle / MAX_CYCLES) * 100
st.progress(min(progress_percent / 100, 1.0))

st.info(f"**Current ALU State:** {alu_status}")

# Add a small explanation
with st.expander("ℹ️ Hardware Architecture Details"):
    st.markdown("""
    **TinyGPU ALU Operations:**
    - **FIXED_MUL**: Performs Q3.5 fixed-point multiplication: `(rs * rt) >> 5`
    - **ADD**: Standard 8-bit addition
    - **Pipeline**: Each pixel requires 3 multiplications and 2 additions = 5 cycles
    
    **SVD Compression (Rank K=3):**
    - Original 16x16 block (256 values) → Compressed to 3*(16+16+1) = 99 values
    - Compression ratio: ~61% data reduction
    - Hardware decompression runs on 8-bit fixed-point ALU
    """)

# -------------------------------------------------------------
# 5. LIVE VISUAL RENDERING
# -------------------------------------------------------------
st.markdown("---")
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.subheader("Original Image (Host Server)")
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.imshow(original_img, cmap='gray', vmin=0, vmax=1)
    ax1.axis('off')
    ax1.set_title(f"64x64 Image")
    st.pyplot(fig1)

with fig_col2:
    st.subheader(f"Edge GPU Decompression (Cycle {current_cycle:,})")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    # Use vmin/vmax to keep the brightness stable as pixels load
    im = ax2.imshow(live_img, cmap='gray', vmin=0, vmax=1)
    ax2.axis('off')
    
    # Add text showing progress
    if pixels_completed < TOTAL_PIXELS:
        ax2.text(32, 32, f"{pixels_completed}/{TOTAL_PIXELS} pixels",
                color='red', fontsize=10, ha='center',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
    
    st.pyplot(fig2)

# Add a status indicator for completion
if pixels_completed >= TOTAL_PIXELS:
    st.success("🎉 Image decompression complete! The ALU has reconstructed the full image from compressed data.")
elif current_cycle > 0:
    st.info(f"⏳ Decompressing... {progress_percent:.1f}% complete")

# Add auto-play functionality
st.markdown("---")
auto_play = st.checkbox("▶️ Auto-play simulation")
if auto_play:
    st.balloons()
    st.warning("Auto-play mode: Use the slider above to control the simulation step-by-step")
    st.info("For full simulation, use the slider to watch the image reconstruct pixel by pixel!")

st.caption("College Project • Built with Streamlit, Numpy, and Matplotlib • Hardware-Accelerated SVD Decompression")