import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import resize

# TinyGPU ALU Opcodes (matches your ALU)
ALU_ADD = 0  # 2'b00
ALU_SUB = 1  # 2'b01
ALU_FIXED_MUL = 2  # 2'b10
ALU_RELU = 3  # 2'b11

FIXED_SCALE = 32  # Q3.5 format (2^5)
FIXED_SCALE_BITS = 5

async def run_alu(dut, op, rs_val, rt_val):
    """ Execute ALU operation in hardware """
    # Clip to 8-bit signed range
    rs_val = max(min(int(rs_val), 127), -128)
    rt_val = max(min(int(rt_val), 127), -128)
    
    # Set inputs
    dut.rs.value = rs_val & 0xFF
    dut.rt.value = rt_val & 0xFF
    dut.decoded_alu_arithmetic_mux.value = op
    
    # Wait for the result (ALU updates on posedge clk)
    await RisingEdge(dut.clk)
    
    # Read result
    raw_out = int(dut.alu_out.value)
    result = raw_out if raw_out < 128 else (raw_out - 256)
    
    return result

@cocotb.test()
async def test_svd_hardware(dut):
    """ Edge AI Decompression: Reconstructing SVD Compressed Images in Silicon """
    
    # Generate clock
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    
    # Reset sequence
    dut.reset.value = 1
    dut.enable.value = 0
    dut.core_state.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Enable the ALU and set core state
    dut.enable.value = 1
    dut.core_state.value = 5  # 3'b101 as per your ALU
    dut.decoded_alu_output_mux.value = 0  # Use arithmetic mode
    
    await RisingEdge(dut.clk)
    
    # Load and prepare image (camera image is already grayscale)
    img = data.camera()
    img = resize(img, (64, 64))
    
    BLOCK_SIZE = 16
    K_RANK = 3
    reconstructed_img = np.zeros((64, 64))
    total_clock_cycles = 0
    
    dut._log.info("=" * 50)
    dut._log.info(f"Software: Compressing image using SVD (Rank {K_RANK})")
    dut._log.info("Hardware: Streaming compressed matrices to ALU")
    dut._log.info("=" * 50)
    
    # Process blocks
    for i in range(0, 64, BLOCK_SIZE):
        for j in range(0, 64, BLOCK_SIZE):
            
            # Extract block and run SVD
            block = img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            U, S, Vt = np.linalg.svd(block, full_matrices=False)
            
            # Keep top K components
            U_k = U[:, :K_RANK]
            S_k = np.diag(S[:K_RANK])
            Vt_k = Vt[:K_RANK, :]
            
            # Pre-multiply US to reduce operations
            US = np.dot(U_k, S_k)
            
            # Quantize to fixed-point
            US_hw = np.clip(np.round(US * FIXED_SCALE), -128, 127).astype(int)
            Vt_hw = np.clip(np.round(Vt_k * FIXED_SCALE), -128, 127).astype(int)
            
            # Hardware decompression
            for r in range(BLOCK_SIZE):
                for c in range(BLOCK_SIZE):
                    # Compute dot product of length K=3
                    m1 = await run_alu(dut, ALU_FIXED_MUL, US_hw[r, 0], Vt_hw[0, c])
                    m2 = await run_alu(dut, ALU_FIXED_MUL, US_hw[r, 1], Vt_hw[1, c])
                    m3 = await run_alu(dut, ALU_FIXED_MUL, US_hw[r, 2], Vt_hw[2, c])
                    total_clock_cycles += 3
                    
                    # Accumulate results
                    a1 = await run_alu(dut, ALU_ADD, m1, m2)
                    final_pixel = await run_alu(dut, ALU_ADD, a1, m3)
                    total_clock_cycles += 2
                    
                    # Convert back to floating point
                    # Hardware does (a * b) >> 5, so we need to divide by 32 twice
                    reconstructed_img[i+r, j+c] = final_pixel / (FIXED_SCALE * FIXED_SCALE)
    
    dut._log.info("=" * 50)
    dut._log.info(f"Hardware Acceleration Complete!")
    dut._log.info(f"Total Clock Cycles: {total_clock_cycles:,}")
    dut._log.info(f"Cycles per Pixel: {total_clock_cycles/(64*64):.1f}")
    dut._log.info("=" * 50)
    
    # Calculate metrics
    mse = np.mean((img - reconstructed_img) ** 2)
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        dut._log.info(f"PSNR: {psnr:.2f} dB")
    
    # Save results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Original 64x64 Image")
    ax1.axis('off')
    
    ax2.imshow(reconstructed_img, cmap='gray')
    ax2.set_title(f"Hardware Decompressed (Rank {K_RANK})\nCycles: {total_clock_cycles:,}")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("SVD_Hardware_Result.png", bbox_inches='tight', dpi=150)
    dut._log.info("SUCCESS: 'SVD_Hardware_Result.png' saved!")

