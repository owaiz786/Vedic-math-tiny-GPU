"""
Step 4: TinyGPU inference kernel for MLP forward pass

Architecture: 16 -> 8 -> 10  (fits in 8-bit data memory, 256 bytes)

Kernel strategy:
  - Layer 1: 8 threads, one per hidden neuron
      each thread: dot(W1[n,:], img) + b1[n], then RELU
  - Layer 2: 10 threads, one per output neuron
      each thread: dot(W2[n,:], hidden) + b2[n]
  - argmax of OUTPUT_BASE[0..9] = predicted digit

ISA reference (from decoder.sv):
  NOP     = 0000  BRnzp  = 0001  CMP  = 0010  ADD  = 0011
  SUB     = 0100  MUL    = 0101  DIV  = 0110  LDR  = 0111
  STR     = 1000  CONST  = 1001  RET  = 1111

  Instruction format (16-bit):
    [15:12] opcode
    [11:8]  Rd  (destination)
    [7:4]   Rs  (source 1)
    [3:0]   Rt  (source 2) / immediate low nibble

  Special registers (read-only):
    0b1100 (%blockIdx)   0b1101 (%blockDim)   0b1110 (%threadIdx)   0b1111 (#0)

ALU opcodes (decoded_alu_arithmetic_mux):
  ADD      = 2'b00  -> vedic_adder      (~5 gate levels)
  SUB      = 2'b01
  FIXED_MUL= 2'b10  -> vedic_multiplier (~6 gate levels, Q3.5 shift fused)
  RELU     = 2'b11  -> hardware ReLU    (1 cycle)

This file produces:
  outputs/kernel_layer1.py  — program memory list for Layer 1 (used by cocotb test)
  outputs/kernel_layer2.py  — program memory list for Layer 2
"""

import os
os.makedirs("outputs", exist_ok=True)

# ── Instruction encoder helpers ───────────────────────────────────────────────

def ADD(rd, rs, rt):    return (0b0011 << 12) | (rd << 8) | (rs << 4) | rt
def SUB(rd, rs, rt):    return (0b0100 << 12) | (rd << 8) | (rs << 4) | rt
def MUL(rd, rs, rt):    return (0b0101 << 12) | (rd << 8) | (rs << 4) | rt
def LDR(rd, rs):        return (0b0111 << 12) | (rd << 8) | (rs << 4)
def STR(rd, rs):        return (0b1000 << 12) | (rd << 8) | (rs << 4)
def CONST(rd, imm):     return (0b1001 << 12) | (rd << 8) | (imm & 0xFF)
def CMP(rs, rt):        return (0b0010 << 12) | (rs << 4) | rt
def BRn(offset):        return (0b0001 << 12) | (0b100 << 8) | (offset & 0xFF)
def RET():              return 0b1111 << 12

# Special register indices
BLOCK_IDX  = 0b1100
BLOCK_DIM  = 0b1101
THREAD_IDX = 0b1110
ZERO       = 0b1111

# Memory base addresses (must match export_mem.py)
IMG_BASE    = 0x000
W1_BASE     = 0x010   # W1: 8 neurons x 16 inputs = 128 bytes
B1_BASE     = 0x090
W2_BASE     = 0x098   # W2: 10 outputs x 8 inputs = 80 bytes
B2_BASE     = 0x0C8
HIDDEN_BASE = 0x0D2
OUTPUT_BASE = 0x0DC

INPUT_DIM  = 16   # 4x4 image
HIDDEN_DIM = 8


# ── Layer 1 kernel ────────────────────────────────────────────────────────────
# Thread n (0..7) computes: hidden[n] = ReLU(dot(W1[n,:], img) + b1[n])
# Registers:
#   R0  = thread global id (n)
#   R1  = loop counter k (0..INPUT_DIM-1)
#   R2  = INPUT_DIM constant (16)
#   R3  = accumulator (acc)
#   R4  = scratch / address
#   R5  = loaded weight W1[n*INPUT_DIM + k]
#   R6  = loaded pixel img[k]
#   R7  = product W1*img
#   R8  = W1_BASE constant
#   R9  = IMG_BASE constant (0, so we skip CONST for it)
#   R10 = B1_BASE constant
#   R11 = HIDDEN_BASE constant

layer1 = [
    # n = blockIdx * blockDim + threadIdx
    MUL(0, BLOCK_IDX, BLOCK_DIM),
    ADD(0, 0, THREAD_IDX),             # R0 = thread id = neuron index n

    CONST(2, INPUT_DIM),               # R2 = 16
    CONST(3, 0),                       # R3 = acc = 0
    CONST(1, 0),                       # R1 = k = 0
    CONST(8, W1_BASE),                 # R8 = W1_BASE
    CONST(10, B1_BASE),                # R10 = B1_BASE
    CONST(11, HIDDEN_BASE),            # R11 = HIDDEN_BASE

    # LOOP (8 instructions):
    #   addr_W = W1_BASE + n*16 + k  = R8 + R0*16 + R1
    MUL(4, 0, 2),                      # R4 = n * INPUT_DIM
    ADD(4, 4, 8),                      # R4 = W1_BASE + n*16
    ADD(4, 4, 1),                      # R4 = W1_BASE + n*16 + k
    LDR(5, 4),                         # R5 = W1[n*16+k]

    #   addr_img = IMG_BASE + k = 0 + k = R1  (IMG_BASE=0, so address IS k)
    LDR(6, 1),                         # R6 = img[k]

    MUL(7, 5, 6),                      # R7 = W1[n,k] * img[k]  (vedic Q3.5)
    ADD(3, 3, 7),                      # acc += product

    ADD(1, 1, ZERO + 1),               # k++  (add 1 using ZERO+1 trick — see note)
    CMP(1, 2),                         # compare k vs INPUT_DIM
    BRn(-9),                           # branch back if k < INPUT_DIM  (9 instrs back)

    # Add bias: acc += b1[n]
    ADD(4, 10, 0),                     # R4 = B1_BASE + n
    LDR(5, 4),                         # R5 = b1[n]
    ADD(3, 3, 5),                      # acc += b1[n]

    # ReLU: if acc[7]==1 (negative) → 0, else acc
    # Use RELU ALU opcode — already hardware in your ALU (alu_arithmetic_mux=2'b11)
    # Encoded as MUL with RELU opcode override via special immediate:
    # In TinyGPU decoder MUL opcode triggers FIXED_MUL; RELU is a separate ALU op.
    # We simulate ReLU with CMP + conditional write:
    CONST(4, 0),                       # R4 = 0
    CMP(3, 4),                         # set NZP: is acc negative?
    # BRn skips the store if negative → store 0 instead
    # Store hidden[n] = max(acc, 0)
    ADD(4, 11, 0),                     # R4 = HIDDEN_BASE + n
    STR(4, 3),                         # hidden[n] = acc  (positive path)
    RET(),

    # TODO: add branch for negative path to store 0
    # For now kernel stores raw accumulator; true RELU requires BRn jump to
    # a "store zero" block — extend this as needed.
]

# ── Layer 2 kernel ────────────────────────────────────────────────────────────
# Thread n (0..9) computes: output[n] = dot(W2[n,:], hidden) + b2[n]
# 10 threads, HIDDEN_DIM=8 inner loop

layer2 = [
    MUL(0, BLOCK_IDX, BLOCK_DIM),
    ADD(0, 0, THREAD_IDX),             # R0 = neuron index n

    CONST(2, HIDDEN_DIM),              # R2 = 8
    CONST(3, 0),                       # R3 = acc = 0
    CONST(1, 0),                       # R1 = k = 0
    CONST(8, W2_BASE),                 # R8 = W2_BASE
    CONST(9, HIDDEN_BASE),             # R9 = HIDDEN_BASE
    CONST(10, B2_BASE),                # R10 = B2_BASE
    CONST(11, OUTPUT_BASE),            # R11 = OUTPUT_BASE

    # LOOP:
    MUL(4, 0, 2),                      # R4 = n * HIDDEN_DIM
    ADD(4, 4, 8),                      # R4 = W2_BASE + n*8
    ADD(4, 4, 1),                      # R4 = W2_BASE + n*8 + k
    LDR(5, 4),                         # R5 = W2[n,k]
    ADD(4, 9, 1),                      # R4 = HIDDEN_BASE + k
    LDR(6, 4),                         # R6 = hidden[k]
    MUL(7, 5, 6),                      # R7 = W2[n,k] * hidden[k]  (vedic Q3.5)
    ADD(3, 3, 7),                      # acc += product
    ADD(1, 1, ZERO + 1),               # k++
    CMP(1, 2),
    BRn(-11),                          # loop back

    # Add bias
    ADD(4, 10, 0),
    LDR(5, 4),
    ADD(3, 3, 5),

    # Store output score
    ADD(4, 11, 0),
    STR(4, 3),
    RET(),
]


def save_kernel(name, program):
    path = f"outputs/kernel_{name}.py"
    with open(path, "w") as f:
        f.write(f"# Auto-generated kernel: {name}\n")
        f.write(f"# {len(program)} instructions\n\n")
        f.write(f"program_{name} = [\n")
        for instr in program:
            f.write(f"    0b{instr:016b},  # 0x{instr:04X}\n")
        f.write("]\n")
    print(f"Saved: {path}")


def print_kernel(name, program):
    print(f"\n{'─'*50}")
    print(f"Kernel: {name}  ({len(program)} instructions)")
    print(f"{'─'*50}")
    for i, instr in enumerate(program):
        print(f"  [{i:02d}]  0b{instr:016b}  0x{instr:04X}")


if __name__ == "__main__":
    save_kernel("layer1", layer1)
    save_kernel("layer2", layer2)
    print_kernel("layer1", layer1)
    print_kernel("layer2", layer2)
    print(f"\nLayer 1 threads needed: {HIDDEN_DIM}")
    print(f"Layer 2 threads needed: 10")
