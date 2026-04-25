import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
import os

# Opcode mappings from our modified alu.sv
ALU_ADD = 0
ALU_SUB = 1
ALU_FIXED_MUL = 2

async def run_alu(dut, op, rs_val, rt_val):
    """ Helper to blast Two's Complement numbers into the Silicon """
    # Convert human numbers to 8-bit hardware logic
    dut.rs.value = rs_val & 0xFF
    dut.rt.value = rt_val & 0xFF
    dut.decoded_alu_arithmetic_mux.value = op
    
    # Wait 1 Clock Cycle for the electrons to flow through the gates
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    
    # Read the output voltage and convert back to human numbers
    raw_out = int(dut.alu_out.value)
    return raw_out if raw_out < 128 else (raw_out - 256)

@cocotb.test()
async def test_mandelbrot(dut):
    """ Hardware-in-the-Loop (HITL) SIMT Fractal Renderer """
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    
    # 1. Boot up the ALU into 'EXECUTE' mode
    dut.reset.value = 1
    dut.enable.value = 0
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    dut.enable.value = 1
    dut.core_state.value = 5 
    dut.decoded_alu_output_mux.value = 0
    
    # 2. Generate the Complex Plane (16x16 pixels)
    width, height = 16, 16
    fixed_scale = 32 # Q3.5 format (32 = 1.0)
    
    pixels =[]
    for y in range(height):
        for x in range(width):
            cx = int((-2.0 + (x/width)*3.0) * fixed_scale)
            cy = int((-1.5 + (y/height)*3.0) * fixed_scale)
            pixels.append((cx, cy))
            
    output_memory =[]
    dut._log.info("==================================================")
    dut._log.info(" STARTING HARDWARE-IN-THE-LOOP FRACTAL RENDERING")
    dut._log.info(" Computing 256 pixels through physical silicon...")
    dut._log.info("==================================================")
    
    # 3. Pump the pixels through the ALU hardware!
    for i, (cx, cy) in enumerate(pixels):
        zx, zy = 0, 0
        iteration = 0
        max_iter = 15
        
        while iteration < max_iter:
            # zx^2 and zy^2 (Using our Custom Q3.5 Hardware Multiplier!)
            zx_sq = await run_alu(dut, ALU_FIXED_MUL, zx, zx)
            zy_sq = await run_alu(dut, ALU_FIXED_MUL, zy, zy)
            
            # Check Escape Condition (zx^2 + zy^2 > 4.0)
            mag_sq = await run_alu(dut, ALU_ADD, zx_sq, zy_sq)
            if mag_sq > 120 or mag_sq < 0: # < 0 catches 8-bit overflow
                break
                
            # temp_zx = zx^2 - zy^2 + cx
            sub_res = await run_alu(dut, ALU_SUB, zx_sq, zy_sq)
            temp_zx = await run_alu(dut, ALU_ADD, sub_res, cx)
            
            # zy = 2 * zx * zy + cy
            zx_zy = await run_alu(dut, ALU_FIXED_MUL, zx, zy)
            two_zx_zy = await run_alu(dut, ALU_ADD, zx_zy, zx_zy)
            zy = await run_alu(dut, ALU_ADD, two_zx_zy, cy)
            
            zx = temp_zx
            iteration += 1
            
        output_memory.append(iteration)
        if i % 32 == 0 and i > 0:
            dut._log.info(f" -> Rendered {i} / 256 pixels...")
            
    # 4. Dump the GPU Memory to a log file for the graphics renderer
    os.makedirs("test/logs", exist_ok=True)
    with open("test/logs/mandelbrot.log", "w") as f:
        for addr, val in enumerate(output_memory):
            f.write(f"Mem[{addr}]: {val}\n")
            
    dut._log.info("==================================================")
    dut._log.info(" FRACTAL COMPLETE! Data written to memory dump.")
    dut._log.info("==================================================")