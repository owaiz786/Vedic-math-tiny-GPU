.PHONY: test compile compile_std compile_vedic compare

export LIBPYTHON_LOC=$(shell cocotb-config --libpython)

# ------------------------------------------------------------------
# Run any test kernel (e.g.  make test_matadd  /  make test_matmul)
# Uses the Vedic ALU by default.
# ------------------------------------------------------------------
test_%:
	make compile
	iverilog -o build/sim.vvp -s gpu -g2012 build/gpu.v
	MODULE=test.test_$* vvp -M $$(cocotb-config --lib-dir)/cocotb/libs -m libcocotbvpi_icarus build/sim.vvp

# ------------------------------------------------------------------
# Run comparison tests: standard ALU vs Vedic ALU
# Saves cycle counts to build/results_std.txt & build/results_vedic.txt
# then prints a side-by-side summary.
# ------------------------------------------------------------------
compare_%:
	@echo "=== Running with STANDARD ALU ==="
	make compile_std
	iverilog -o build/sim_std.vvp -s gpu -g2012 build/gpu_std.v
	MODULE=test.test_$* RESULT_FILE=build/results_std.txt \
	    vvp -M $$(cocotb-config --lib-dir) -m libcocotbvpi_icarus build/sim_std.vvp \
	    2>&1 | tee build/log_std.txt
	@echo ""
	@echo "=== Running with VEDIC ALU ==="
	make compile_vedic
	iverilog -o build/sim_vedic.vvp -s gpu -g2012 build/gpu_vedic.v
	MODULE=test.test_$* RESULT_FILE=build/results_vedic.txt \
	    vvp -M $$(cocotb-config --lib-dir) -m libcocotbvpi_icarus build/sim_vedic.vvp \
	    2>&1 | tee build/log_vedic.txt
	@echo ""
	@echo "=== Comparison Summary ==="
	python3 test/compare_results.py build/log_std.txt build/log_vedic.txt

# ------------------------------------------------------------------
# Compile using VEDIC ALU (default)
# ------------------------------------------------------------------
compile: compile_vedic_alu
	sv2v $$(find src -name "*.sv" ! -name "alu_standard.sv" ! -name "alu.sv" ! -name "vedic_adder.sv" ! -name "vedic_multiplier.sv") -w build/gpu.v
	echo "" >> build/gpu.v
	cat build/vedic_adder.v    >> build/gpu.v
	cat build/vedic_multiplier.v >> build/gpu.v
	cat build/alu_vedic.v      >> build/gpu.v
	echo '`timescale 1ns/1ns' > build/temp.v
	cat build/gpu.v >> build/temp.v
	mv build/temp.v build/gpu.v

# Compile GPU using STANDARD ALU (for comparison)
compile_std: compile_std_alu
	sv2v $$(find src -name "*.sv" ! -name "alu_standard.sv" ! -name "alu.sv" ! -name "vedic_adder.sv" ! -name "vedic_multiplier.sv") -w build/gpu_std.v
	echo "" >> build/gpu_std.v
	cat build/alu_std.v >> build/gpu_std.v
	echo '`timescale 1ns/1ns' > build/temp.v
	cat build/gpu_std.v >> build/temp.v
	mv build/temp.v build/gpu_std.v

# Compile GPU using VEDIC ALU (for comparison)
compile_vedic: compile_vedic_alu
	sv2v $$(find src -name "*.sv" ! -name "alu_standard.sv" ! -name "alu.sv" ! -name "vedic_adder.sv" ! -name "vedic_multiplier.sv") -w build/gpu_vedic.v
	echo "" >> build/gpu_vedic.v
	cat build/vedic_adder.v      >> build/gpu_vedic.v
	cat build/vedic_multiplier.v >> build/gpu_vedic.v
	cat build/alu_vedic.v        >> build/gpu_vedic.v
	echo '`timescale 1ns/1ns' > build/temp.v
	cat build/gpu_vedic.v >> build/temp.v
	mv build/temp.v build/gpu_vedic.v

# ------------------------------------------------------------------
# Individual module compilation
# ------------------------------------------------------------------
compile_std_alu:
	sv2v src/alu_standard.sv -w build/alu_std.v

compile_vedic_alu:
	sv2v src/vedic_adder.sv      -w build/vedic_adder.v
	sv2v src/vedic_multiplier.sv -w build/vedic_multiplier.v
	sv2v src/alu.sv              -w build/alu_vedic.v

compile_%:
	sv2v -w build/$*.v src/$*.sv

show_%: %.vcd %.gtkw
	gtkwave $^