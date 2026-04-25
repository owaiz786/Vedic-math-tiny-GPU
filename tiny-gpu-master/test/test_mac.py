import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge

@cocotb.test()
async def test_mac(dut):
    """ Hardware Unit Test for Custom AI Instruction (MAC) """
    
    # 1. Start a 10ns hardware clock
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # 2. Initialize the ALU and clear the Accumulator (Reset = 1)
    dut.reset.value = 1
    dut.enable.value = 0
    dut.rt.value = 0
    dut.rs.value = 0
    await RisingEdge(dut.clk)
    
    # 3. Boot up the ALU into 'EXECUTE' mode
    dut.reset.value = 0
    dut.enable.value = 1
    dut.core_state.value = 5  
    
    # 4. Send our custom 'MAC' Opcode (2'b10)
    dut.decoded_alu_output_mux.value = 0 
    dut.decoded_alu_arithmetic_mux.value = 2 

    # 5. Define our dot product inputs (Weight * Input)
    # We will compute: (2 * 3) + (4 * 5) + (1 * 10) = 36
    inputs_rs =[2, 4, 1]  # Imagine these are Neural Network Weights
    inputs_rt =[3, 5, 10] # Imagine these are Input Pixels
    expected_outputs =[6, 26, 36]

    dut._log.info("==================================================")
    dut._log.info(" STARTING HARDWARE MAC (TENSOR CORE) VERIFICATION")
    dut._log.info("==================================================")

    for i in range(len(inputs_rs)):
        val_rs = inputs_rs[i]
        val_rt = inputs_rt[i]
        
        # Inject the Weight into 'rs' and the Input into 'rt'
        dut.rs.value = val_rs
        dut.rt.value = val_rt
        
        # Wait for EXACTLY 1 Clock Cycle
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk) 
        
        out_val = int(dut.alu_out.value)
        
        dut._log.info(f" CLOCK CYCLE {i+1} | Multiply {val_rs:2} * {val_rt:2} | Running Total (Accumulator): {out_val:3}")
        
        assert out_val == expected_outputs[i], f"Hardware Failed!"

    dut._log.info("==================================================")
    dut._log.info(" SUCCESS: DOT PRODUCT COMPUTED IN 1 CYCLE PER MULTIPLY!")
    dut._log.info("==================================================")
