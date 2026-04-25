#!/bin/bash

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source cocotb_venv/bin/activate
fi

# Set all required environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOPLEVEL=alu
export MODULE=test.test_svd
export COCOTB_TEST_MODULES=test.test_svd
export PYGPI_PYTHON_BIN=$(which python)

# Get Python library path
PYTHON_LIB=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export LD_LIBRARY_PATH=$PYTHON_LIB:$LD_LIBRARY_PATH

# Get cocotb libs path
COCOTB_LIBS=$(python -c "import cocotb; import os; print(os.path.dirname(cocotb.__file__) + '/libs')")

echo "========================================="
echo "Environment variables:"
echo "PYTHONPATH: $PYTHONPATH"
echo "TOPLEVEL: $TOPLEVEL"
echo "MODULE: $MODULE"
echo "COCOTB_TEST_MODULES: $COCOTB_TEST_MODULES"
echo "PYGPI_PYTHON_BIN: $PYGPI_PYTHON_BIN"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "COCOTB_LIBS: $COCOTB_LIBS"
echo "========================================="

# Check if test file exists
if [ ! -f "test/test_svd.py" ]; then
    echo "❌ ERROR: test/test_svd.py not found!"
    exit 1
fi

# Check if test/__init__.py exists
if [ ! -f "test/__init__.py" ]; then
    echo "Creating test/__init__.py"
    touch test/__init__.py
fi

echo "✅ Test file found: test/test_svd.py"
echo "========================================="
echo "Starting simulation..."
echo "========================================="

# Run the simulation
vvp -M $COCOTB_LIBS -m libcocotbvpi_icarus build/sim_alu.vvp

if [ $? -eq 0 ]; then
    echo "========================================="
    echo "✅ Simulation completed successfully!"
    echo "========================================="
else
    echo "========================================="
    echo "❌ Simulation failed with error code $?"
    echo "========================================="
fi
