`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// VEDIC ALU  —  drop-in replacement for alu.sv
// =============================================================================
// Changes vs the original alu.sv:
//
//   ADD case  → uses vedic_adder   (carry-select,  Anurupyena sutra)
//               Critical path: ~5 gate levels  vs ~8 ripple-carry
//
//   FIXED_MUL → uses vedic_multiplier (Urdhva-Tiryakbhyam sutra)
//               Critical path: ~6 gate levels  vs ~16 array multiplier
//               The Q3.5 >>>5 shift is absorbed into the multiplier wiring.
//
//   SUB, CMP, RELU — unchanged (SUB still uses - operator; Vedic subtraction
//               uses the same carry-select trick but sub is rarely on the
//               critical path in these kernels — left as a future exercise).
//
// The module interface is IDENTICAL to alu.sv so core.sv needs no changes.
// =============================================================================

module alu (
    input wire clk,
    input wire reset,
    input wire enable,

    input reg [2:0] core_state,
    input reg [1:0] decoded_alu_arithmetic_mux,
    input reg       decoded_alu_output_mux,

    input  reg [7:0] rs,
    input  reg [7:0] rt,
    output wire [7:0] alu_out
);
    localparam ADD       = 2'b00,
               SUB       = 2'b01,
               FIXED_MUL = 2'b10,   // Q3.5 Fixed-Point Multiplier (Mandelbrot)
               RELU      = 2'b11;   // Custom ReLU for AI kernels

    reg  [7:0] alu_out_reg;
    assign alu_out = alu_out_reg;

    // -----------------------------------------------------------------------
    // Vedic adder — instantiated combinatorially outside the always block
    // so the full carry-select tree is synthesized as pure combinational logic.
    // -----------------------------------------------------------------------
    wire [7:0] vedic_sum;
    wire       vedic_carry;   // available for future overflow detection

    vedic_adder u_adder (
        .a   (rs),
        .b   (rt),
        .sum (vedic_sum),
        .cout(vedic_carry)
    );

    // -----------------------------------------------------------------------
    // Vedic multiplier — combinatorial, Q3.5 shift fused into bit-select.
    // -----------------------------------------------------------------------
    wire signed [7:0] vedic_product;

    vedic_multiplier u_mul (
        .a           (rs),
        .b           (rt),
        .product_q35 (vedic_product)
    );

    // -----------------------------------------------------------------------
    // Registered output — same clocking structure as original alu.sv
    // -----------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset) begin
            alu_out_reg <= 8'b0;

        end else if (enable) begin
            if (core_state == 3'b101) begin

                if (decoded_alu_output_mux == 1) begin
                    // ---------------------------------------------------------
                    // SIGNED COMPARISON — unchanged from original
                    // Required by Mandelbrot for negative-number branching.
                    // ---------------------------------------------------------
                    alu_out_reg <= {5'b0,
                                    ($signed(rs) > $signed(rt)),
                                    (rs == rt),
                                    ($signed(rs) < $signed(rt))};

                end else begin
                    case (decoded_alu_arithmetic_mux)

                        // -----------------------------------------------------
                        // ADD: Vedic carry-select (Anurupyena)
                        // Replaces: rs + rt  (ripple-carry, 8 gate levels)
                        // With:     vedic_sum (carry-select, ~5 gate levels)
                        // -----------------------------------------------------
                        ADD: alu_out_reg <= vedic_sum;

                        // SUB: unchanged — Vedic borrow-select is future work
                        SUB: alu_out_reg <= rs - rt;

                        // -----------------------------------------------------
                        // FIXED_MUL: Vedic Urdhva-Tiryakbhyam
                        // Replaces: ($signed(rs) * $signed(rt)) >>> 5
                        //           (array multiplier, ~16 gate levels)
                        // With:     vedic_product  (~6 gate levels, shift fused)
                        // -----------------------------------------------------
                        FIXED_MUL: alu_out_reg <= vedic_product;

                        // RELU: unchanged
                        RELU: alu_out_reg <= (rs[7] == 1'b1) ? 8'b0 : rs;

                    endcase
                end
            end
        end
    end

endmodule