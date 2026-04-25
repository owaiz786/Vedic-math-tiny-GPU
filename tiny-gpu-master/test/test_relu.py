import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge

@cocotb.test()
async def test_relu(dut):
    """ Hardware Unit Test for Custom AI Instruction (ReLU) """
    
    # 1. Start a 10ns hardware clock
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # 2. Initialize the ALU control wires
    dut.reset.value = 1
    dut.enable.value = 0
    dut.rt.value = 0
    await RisingEdge(dut.clk)
    
    # 3. Boot up the ALU into 'EXECUTE' mode
    dut.reset.value = 0
    dut.enable.value = 1
    dut.core_state.value = 5  # 3'b101 is the EXECUTE state in TinyGPU
    
    # 4. Send our custom 'RELU' Opcode (2'b11) to the instruction decoder wires
    dut.decoded_alu_output_mux.value = 0 
    dut.decoded_alu_arithmetic_mux.value = 3 

    # 5. Define our Test Neural Network Data (Negative and Positive numbers)
    test_inputs =[-10, 25, -3, 42]
    expected_outputs = [0, 25, 0, 42]

    dut._log.info("==================================================")
    dut._log.info(" STARTING HARDWARE RELU VERIFICATION")
    dut._log.info("==================================================")

    for i in range(len(test_inputs)):
        in_val = test_inputs[i]
        
        # Convert human integers into 8-bit Silicon binary (Two's Complement)
        binary_val = in_val if in_val >= 0 else (256 + in_val)
        
        # Inject the value into the GPU's 'rs' register
        dut.rs.value = binary_val
        
        # Wait for exactly 1 Clock Cycle for the electrons to flow
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk) 
        
        # Read the resulting voltage from the output wire and convert back to int
        raw_out = int(dut.alu_out.value)
        out_val = raw_out if raw_out < 128 else (raw_out - 256)
        
        dut._log.info(f" CLOCK CYCLE {i+1} | Input: {in_val:3}  ->  Hardware Output: {out_val:3}")
        
        assert out_val == expected_outputs[i], f"Hardware Failed on input {in_val}"

    dut._log.info("==================================================")
    dut._log.info(" SUCCESS: ALL MATH COMPLETED IN 1 CYCLE PER INPUT!")
    dut._log.info("==================================================")
