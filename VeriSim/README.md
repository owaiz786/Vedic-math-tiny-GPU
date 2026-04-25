# VeriSim

VeriSim is a Verilog simulator with integrated waveform visualization. Write, simulate, and visualize your Verilog designs in a modern, user-friendly interface.

## Features
- Dual-pane editor with Verilog syntax highlighting
- Real-time waveform visualization
- Dark theme optimized interface
- Automatic VCD generation
- Downloadable waveform visualizations

## Prerequisites

Before installing VeriSim, ensure you have:
- Python 3.8 or higher
- Icarus Verilog (iverilog)

For macOS users, you can install Icarus Verilog using Homebrew:
```bash
brew install icarus-verilog
```

## Installation

1. Go to home directory and clone the repository:
```bash
cd ~
git clone https://github.com/thevoxium/VeriSim.git
cd VeriSim
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Make VeriSim accessible from anywhere (macOS):
```bash
# Create an alias in your ~/.zshrc or ~/.bash_profile
echo 'alias verisim="cd ~/VeriSim && streamlit run visualise.py"' >> ~/.zshrc

# Reload your shell configuration
source ~/.zshrc
```

## Usage

After installation, you can start VeriSim by typing in your terminal:
```bash
verisim
```

This will open VeriSim in your default web browser.

## Writing Your First Simulation

1. Write your design code in the right editor
2. Write your testbench in the left editor
3. Click "Run Simulation" to see the waveforms
4. VCD dump commands are automatically handled by VeriSim

## License

MIT

