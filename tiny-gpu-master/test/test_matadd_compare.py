"""
test_matadd_compare.py
======================
Matrix addition kernel — instrumented for Standard vs Vedic ALU comparison.

The kernel is identical to test_matadd.py.  The difference is that this
file prints a structured summary line that compare_results.py can parse:

    KERNEL_RESULT: matadd | alu=<std|vedic> | cycles=<N> | correct=<yes|no>

Usage (via Makefile):
    make compare_matadd
"""

import os
import cocotb
from cocotb.triggers import RisingEdge
from .helpers.setup import setup
from .helpers.memory import Memory
from .helpers.logger import logger

ALU_LABEL = os.environ.get("ALU_TYPE", "unknown")   # set by Makefile: std / vedic


@cocotb.test()
async def test_matadd(dut):
    # -----------------------------------------------------------------------
    # Program: C[i] = A[i] + B[i]  for i in 0..7   (uses ADD and LDR/STR)
    # -----------------------------------------------------------------------
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    program = [
        0b0101000011011110,  # MUL R0, %blockIdx, %blockDim
        0b0011000000001111,  # ADD R0, R0, %threadIdx         ; i = blockIdx*blockDim + threadIdx
        0b1001000100000000,  # CONST R1, #0                   ; baseA
        0b1001001000001000,  # CONST R2, #8                   ; baseB
        0b1001001100010000,  # CONST R3, #16                  ; baseC
        0b0011010000010000,  # ADD R4, R1, R0                 ; addr(A[i])
        0b0111010001000000,  # LDR R4, R4                     ; load A[i]
        0b0011010100100000,  # ADD R5, R2, R0                 ; addr(B[i])
        0b0111010101010000,  # LDR R5, R5                     ; load B[i]
        0b0011011001000101,  # ADD R6, R4, R5                 ; C[i] = A[i]+B[i]  ← ALU ADD
        0b0011011100110000,  # ADD R7, R3, R0                 ; addr(C[i])
        0b1000000001110110,  # STR R7, R6                     ; store C[i]
        0b1111000000000000,  # RET
    ]

    data_memory = Memory(dut=dut, addr_bits=8, data_bits=8, channels=4, name="data")
    data = [
        0, 1, 2, 3, 4, 5, 6, 7,   # Matrix A (1×8)
        0, 1, 2, 3, 4, 5, 6, 7,   # Matrix B (1×8)
    ]
    threads = 8

    await setup(
        dut=dut,
        program_memory=program_memory,
        program=program,
        data_memory=data_memory,
        data=data,
        threads=threads,
    )

    data_memory.display(24)

    cycles = 0
    while dut.done.value != 1:
        data_memory.run()
        program_memory.run()
        await cocotb.triggers.ReadOnly()
        await RisingEdge(dut.clk)
        cycles += 1

    data_memory.display(24)

    # Verify results
    expected = [a + b for a, b in zip(data[0:8], data[8:16])]
    correct = True
    for i, exp in enumerate(expected):
        got = data_memory.memory[i + 16]
        if got != exp:
            correct = False
            logger.error(f"  mismatch at C[{i}]: expected {exp}, got {got}")

    status = "yes" if correct else "no"
    logger.info(
        f"KERNEL_RESULT: matadd | alu={ALU_LABEL} | cycles={cycles} | correct={status}"
    )
    logger.info(f"Completed in {cycles} cycles  [ALU={ALU_LABEL}]")

    assert correct, "Matrix addition results do not match expected values"