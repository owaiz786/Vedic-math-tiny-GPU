.PHONY: clean compile compile_std compile_vedic compare test_% inference

# ------------------------------------------------------------------
# Build the complete GPU Verilog file from SystemVerilog sources
# ------------------------------------------------------------------
compile: compile_vedic_alu
	mkdir -p build
	sv2v $$(find src -name "*.sv" ! -name "alu_standard.sv" ! -name "alu.sv" ! -name "vedic_adder.sv" ! -name "vedic_multiplier.sv") -w build/gpu.v
	echo "" >> build/gpu.v
	cat build/vedic_adder.v      >> build/gpu.v
	cat build/vedic_multiplier.v >> build/gpu.v
	cat build/alu_vedic.v        >> build/gpu.v
	echo '`timescale 1ns/1ns' > build/temp.v
	cat build/gpu.v >> build/temp.v
	mv build/temp.v build/gpu.v
	@echo "Compiling with iverilog..."
	iverilog -o build/sim.vvp -s gpu -g2012 build/gpu.v

# ------------------------------------------------------------------
# Compile using STANDARD ALU (for comparison)
# ------------------------------------------------------------------
compile_std: compile_std_alu
	mkdir -p build
	sv2v $$(find src -name "*.sv" ! -name "alu_standard.sv" ! -name "alu.sv" ! -name "vedic_adder.sv" ! -name "vedic_multiplier.sv") -w build/gpu_std.v
	echo "" >> build/gpu_std.v
	cat build/alu_std.v >> build/gpu_std.v
	echo '`timescale 1ns/1ns' > build/temp.v
	cat build/gpu_std.v >> build/temp.v
	mv build/temp.v build/gpu_std.v
	iverilog -o build/sim_std.vvp -s gpu -g2012 build/gpu_std.v

# ------------------------------------------------------------------
# Compile using VEDIC ALU (for comparison target)
# ------------------------------------------------------------------
compile_vedic: compile_vedic_alu
	mkdir -p build
	sv2v $$(find src -name "*.sv" ! -name "alu_standard.sv" ! -name "alu.sv" ! -name "vedic_adder.sv" ! -name "vedic_multiplier.sv") -w build/gpu_vedic.v
	echo "" >> build/gpu_vedic.v
	cat build/vedic_adder.v      >> build/gpu_vedic.v
	cat build/vedic_multiplier.v >> build/gpu_vedic.v
	cat build/alu_vedic.v        >> build/gpu_vedic.v
	echo '`timescale 1ns/1ns' > build/temp.v
	cat build/gpu_vedic.v >> build/temp.v
	mv build/temp.v build/gpu_vedic.v
	iverilog -o build/sim_vedic.vvp -s gpu -g2012 build/gpu_vedic.v

# ------------------------------------------------------------------
# Individual module compilation
# ------------------------------------------------------------------
compile_std_alu:
	mkdir -p build
	sv2v src/alu_standard.sv -w build/alu_std.v

compile_vedic_alu:
	mkdir -p build
	sv2v src/vedic_adder.sv      -w build/vedic_adder.v
	sv2v src/vedic_multiplier.sv -w build/vedic_multiplier.v
	sv2v src/alu.sv              -w build/alu_vedic.v

compile_%:
	mkdir -p build
	sv2v -w build/$*.v src/$*.sv

# ------------------------------------------------------------------
# Run any test kernel
#   make test_matadd / make test_matmul / make test_inference
# ------------------------------------------------------------------
test_%: compile
	COCOTB_TEST_MODULES=test.test_$* \
	vvp -M $$(cocotb-config --lib-dir) -m libcocotbvpi_icarus build/sim.vvp

# ------------------------------------------------------------------
# Run MNIST inference simulation specifically
#   make inference
# ------------------------------------------------------------------
inference: compile
	COCOTB_TEST_MODULES=test.test_inference \
	vvp -M $$(cocotb-config --lib-dir) -m libcocotbvpi_icarus build/sim.vvp

# ------------------------------------------------------------------
# Comparison: standard ALU vs Vedic ALU
#   make compare_matmul / make compare_matadd / make compare_inference
# ------------------------------------------------------------------
compare_%:
	@echo "=== Running with STANDARD ALU ==="
	$(MAKE) compile_std
	COCOTB_TEST_MODULES=test.test_$* RESULT_FILE=build/results_std.txt \
	    vvp -M $$(cocotb-config --lib-dir) -m libcocotbvpi_icarus build/sim_std.vvp \
	    2>&1 | tee build/log_std.txt
	@echo ""
	@echo "=== Running with VEDIC ALU ==="
	$(MAKE) compile_vedic
	COCOTB_TEST_MODULES=test.test_$* RESULT_FILE=build/results_vedic.txt \
	    vvp -M $$(cocotb-config --lib-dir) -m libcocotbvpi_icarus build/sim_vedic.vvp \
	    2>&1 | tee build/log_vedic.txt
	@echo ""
	@echo "=== Comparison Summary ==="
	python3 test/compare_results.py build/log_std.txt build/log_vedic.txt

# ------------------------------------------------------------------
# Clean build directory
# ------------------------------------------------------------------
clean:
	rm -rf build/
	rm -f results.xml
	rm -f *.vcd
	rm -f *.vvp

# ------------------------------------------------------------------
# Show waveform (if you have GTKWave)
# ------------------------------------------------------------------
show_%: %.vcd %.gtkw
	gtkwave $^