`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// STANDARD ALU  —  original implementation (ripple-carry add, array multiply)
// Kept verbatim for comparison against vedic alu.sv.
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
               FIXED_MUL = 2'b10,   // Q3.5 Fixed-Point Multiplier for Mandelbrot
               RELU      = 2'b11;   // Custom ReLU for AI

    reg [7:0] alu_out_reg;
    assign alu_out = alu_out_reg;

    always @(posedge clk) begin
        if (reset) begin
            alu_out_reg <= 8'b0;

        end else if (enable) begin
            if (core_state == 3'b101) begin
                if (decoded_alu_output_mux == 1) begin
                    // Signed comparison for Mandelbrot branching
                    alu_out_reg <= {5'b0,
                                    ($signed(rs) > $signed(rt)),
                                    (rs == rt),
                                    ($signed(rs) < $signed(rt))};
                end else begin
                    case (decoded_alu_arithmetic_mux)
                        // Standard ripple-carry adder (~8 gate levels)
                        ADD: alu_out_reg <= rs + rt;

                        SUB: alu_out_reg <= rs - rt;

                        // Array multiplier + explicit >>>5 shift (~16+1 gate levels)
                        FIXED_MUL: begin
                            alu_out_reg <= ($signed(rs) * $signed(rt)) >>> 5;
                        end

                        // ReLU
                        RELU: begin
                            alu_out_reg <= (rs[7] == 1'b1) ? 8'b0 : rs;
                        end
                    endcase
                end
            end
        end
    end
endmodule