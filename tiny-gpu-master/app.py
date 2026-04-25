import streamlit as st
import pandas as pd
import re
import os
import glob

# -------------------------------------------------------------
# 1. PAGE SETUP & STYLING
# -------------------------------------------------------------
st.set_page_config(page_title="TinyGPU Visualizer", layout="wide", page_icon="🚀")
st.title("🚀 Advanced TinyGPU Telemetry Dashboard")
st.markdown("Real-time Regex parsing of hardware execution logs into visual analytics.")

# -------------------------------------------------------------
# 2. FIND AND LOAD THE LATEST LOG FILE
# -------------------------------------------------------------
log_dir = "test/logs/"
log_files = glob.glob(os.path.join(log_dir, "*.txt"))

if not log_files:
    st.error("❌ No log files found in test/logs/. Please run 'make test_matadd' first!")
    st.stop()

latest_log = max(log_files, key=os.path.getctime)

with open(latest_log, "r") as f:
    raw_text = f.read()

# Chunk the log into Execution Steps
log_blocks = raw_text.split('\n\n')
blocks = [b for b in log_blocks if len(b.strip()) > 10]

if not blocks:
    st.warning("Log file seems empty or format is unexpected.")
    st.stop()

# -------------------------------------------------------------
# 3. TIME TRAVEL SLIDER
# -------------------------------------------------------------
st.markdown("### ⏱️ Hardware Clock Cycle Control")
step = st.slider("Drag to step through GPU Execution time:", 
                 min_value=0, max_value=len(blocks)-1, value=0, step=1)

current_block = blocks[step]

# -------------------------------------------------------------
# 4. REGEX PARSER ALGORITHM
# -------------------------------------------------------------
# We use Regex to hunt for lines that look like "Address | Value" or "Reg: Value"
# Pattern explained: 
# (\d+)      -> Captures the Address/Register number
# \s*[|:=]\s* -> Looks for a separator like '|', ':', or '=' surrounded by spaces
# (-?\d+)    -> Captures the Value (including negative numbers!)
def parse_telemetry(text_block):
    parsed_data =[]
    lines = text_block.split('\n')
    
    for line in lines:
        match = re.search(r'(\d+)\s*[|:=]\s*(-?\d+)', line)
        if match:
            address = f"Loc {match.group(1)}"
            value = int(match.group(2))
            parsed_data.append({"Location": address, "Value": value})
            
    return pd.DataFrame(parsed_data)

df = parse_telemetry(current_block)

# -------------------------------------------------------------
# 5. UI RENDERING (THREE COLUMNS)
# -------------------------------------------------------------
st.markdown("---")
col1, col2, col3 = st.columns([1.5, 1, 1.5])

# Left Column: The Raw Hardware Output
with col1:
    st.subheader("⚙️ Raw Hardware Trace")
    st.code(current_block, language="text")

# Middle Column: The Extracted Pandas DataFrame
with col2:
    st.subheader("📊 Extracted Memory State")
    if not df.empty:
        # We apply a colored heatmap to the table using Pandas!
        styled_df = df.style.background_gradient(cmap='viridis', subset=['Value'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No numeric memory/register data detected in this specific clock cycle.")

# Right Column: The Animated Bar Chart
with col3:
    st.subheader("📈 Visual State Machine")
    if not df.empty:
        # Set the X-axis to the Location, and graph the Values
        chart_data = df.set_index("Location")
        st.bar_chart(chart_data, color="#00ff00")
    else:
        st.write("Awaiting numeric data to render chart...")

st.markdown("---")
st.caption("College Project • Built with Streamlit, Pandas, and Python Regex")

