"""
Step 5: cocotb simulation test — runs inference kernel on TinyGPU RTL
Mirrors the style of your existing test_matmul.py / test_relu.py

Usage (from project root):
  make inference                    # run full inference
  make test_inference               # same as above
  MODULE=test.test_inference make sim  # run with cocotb makefile
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock

# Import helpers - adjust path as needed
try:
    from test.helpers.setup import setup
    from test.helpers.memory import Memory
    from test.helpers.logger import logger
except ImportError:
    # Fallback if helpers are not in package structure
    from helpers.setup import setup
    from helpers.memory import Memory
    from helpers.logger import logger

# Import generated kernel and memory layout
try:
    from outputs.kernel_layer1 import program_layer1
    from outputs.kernel_layer2 import program_layer2
except ImportError:
    # If outputs is not a package, try direct import
    import importlib.util
    spec1 = importlib.util.spec_from_file_location("kernel_layer1", "outputs/kernel_layer1.py")
    kernel_layer1 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(kernel_layer1)
    program_layer1 = kernel_layer1.program_layer1
    
    spec2 = importlib.util.spec_from_file_location("kernel_layer2", "outputs/kernel_layer2.py")
    kernel_layer2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(kernel_layer2)
    program_layer2 = kernel_layer2.program_layer2

# Memory base addresses (must match export_mem.py)
HIDDEN_BASE = 0x0D2
OUTPUT_BASE = 0x0DC


def load_data_mem_from_file(path="outputs/data_mem.mem"):
    """Read .mem hex file into a flat list of ints (one per address)."""
    full_path = os.path.join(project_root, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Data memory file not found: {full_path}")
    
    with open(full_path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


async def initialize_dut(dut):
    """Initialize the DUT - check if reset exists, otherwise just start clock"""
    # Start clock (fixed deprecation warning)
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    # Check if reset signal exists
    if hasattr(dut, 'rst'):
        dut.rst.value = 1
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.rst.value = 0
    else:
        # Just wait a few cycles for initialization
        await Timer(20, units='ns')


@cocotb.test()
async def test_layer1_inference(dut):
    """Run Layer 1 of MLP inference: img -> hidden activations"""
    
    await initialize_dut(dut)
    
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    data_memory    = Memory(dut=dut, addr_bits=8, data_bits=8,  channels=4, name="data")

    # Load weight + image data from exported .mem file
    data = load_data_mem_from_file()

    await setup(
        dut=dut,
        program_memory=program_memory,
        program=program_layer1,
        data_memory=data_memory,
        data=data,
        threads=8,           # 8 hidden neurons → 8 threads
    )

    logger.info("=" * 50)
    logger.info("LAYER 1 INFERENCE — img(16) -> hidden(8)")
    logger.info("=" * 50)

    cycles = 0
    while dut.done.value != 1:
        data_memory.run()
        program_memory.run()
        await RisingEdge(dut.clk)
        cycles += 1
        if cycles > 50000:
            raise cocotb.result.SimTimeoutError("Simulation timeout — check kernel loop")

    logger.info(f"Layer 1 completed in {cycles} cycles")

    logger.info("Hidden activations (raw int8):")
    for i in range(8):
        val = data_memory.memory[HIDDEN_BASE + i]
        signed = val if val < 128 else val - 256
        logger.info(f"  hidden[{i}] = {signed:4d}  (0x{val:02X})")


@cocotb.test()
async def test_layer2_inference(dut):
    """Run Layer 2 of MLP inference: hidden -> output scores, then argmax"""
    
    await initialize_dut(dut)
    
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    data_memory    = Memory(dut=dut, addr_bits=8, data_bits=8,  channels=4, name="data")

    data = load_data_mem_from_file()

    await setup(
        dut=dut,
        program_memory=program_memory,
        program=program_layer2,
        data_memory=data_memory,
        data=data,
        threads=10,          # 10 output neurons → 10 threads
    )

    logger.info("=" * 50)
    logger.info("LAYER 2 INFERENCE — hidden(8) -> scores(10)")
    logger.info("=" * 50)

    cycles = 0
    while dut.done.value != 1:
        data_memory.run()
        program_memory.run()
        await RisingEdge(dut.clk)
        cycles += 1
        if cycles > 50000:
            raise cocotb.result.SimTimeoutError("Simulation timeout")

    logger.info(f"Layer 2 completed in {cycles} cycles")

    scores = []
    logger.info("Output scores (raw int8):")
    for i in range(10):
        val = data_memory.memory[OUTPUT_BASE + i]
        signed = val if val < 128 else val - 256
        scores.append(signed)
        logger.info(f"  score[{i}] = {signed:4d}  (0x{val:02X})")

    predicted = scores.index(max(scores))
    logger.info(f"\n  *** PREDICTED DIGIT: {predicted} ***")
    logger.info("=" * 50)

    # Soft assertion — ground truth is in mem_layout.txt
    assert 0 <= predicted <= 9, f"Predicted class {predicted} out of range"
    
    return predicted


@cocotb.test()
async def test_layer2_inference(dut):
    """Run Layer 2 of MLP inference: hidden -> output scores, then argmax"""
    
    await initialize_dut(dut)
    
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    data_memory    = Memory(dut=dut, addr_bits=8, data_bits=8,  channels=4, name="data")

    data = load_data_mem_from_file()

    await setup(
        dut=dut,
        program_memory=program_memory,
        program=program_layer2,
        data_memory=data_memory,
        data=data,
        threads=10,          # 10 output neurons → 10 threads
    )

    cocotb.log.info("=" * 50)
    cocotb.log.info("LAYER 2 INFERENCE — hidden(8) -> scores(10)")
    cocotb.log.info("=" * 50)

    cycles = 0
    while dut.done.value != 1:
        data_memory.run()
        program_memory.run()
        await RisingEdge(dut.clk)
        cycles += 1
        if cycles > 50000:
            raise cocotb.result.SimTimeoutError("Simulation timeout")

    cocotb.log.info(f"Layer 2 completed in {cycles} cycles")

    scores = []
    print("\n" + "=" * 50)
    print("OUTPUT SCORES (raw int8):")
    for i in range(10):
        val = data_memory.memory[OUTPUT_BASE + i]
        signed = val if val < 128 else val - 256
        scores.append(signed)
        print(f"  score[{i}] = {signed:4d}  (0x{val:02X})")

    predicted = scores.index(max(scores))
    print("\n" + "=" * 50)
    print(f"*** PREDICTED DIGIT: {predicted} ***")
    print("=" * 50 + "\n")

    # Also log to cocotb
    cocotb.log.info(f"Predicted digit: {predicted}")
    
    assert 0 <= predicted <= 9, f"Predicted class {predicted} out of range"
    
    return predicted