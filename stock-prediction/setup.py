#!/usr/bin/env python3
"""
setup.py — Auto-installer for Stock Prediction + SHAP
Run once before launching: python setup.py
"""

import subprocess
import sys
import os

def run(cmd):
    print(f"  → {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def step(msg):
    print(f"\n{'='*55}\n  {msg}\n{'='*55}")

step("1/3  Checking Python version")
major, minor = sys.version_info[:2]
if major < 3 or minor < 9:
    print(f"  ❌ Python 3.9+ required. You have {major}.{minor}")
    sys.exit(1)
print(f"  ✅ Python {major}.{minor} — OK")

step("2/3  Installing Python packages")
run(f"{sys.executable} -m pip install --upgrade pip")
run(f"{sys.executable} -m pip install -r requirements.txt")
print("  ✅ Packages installed")

step("3/3  Creating data directories")
os.makedirs("data/raw", exist_ok=True)
print("  ✅ data/raw/ created (stock CSVs will be cached here)")

print(f"""
{'='*55}
  ✅ Setup complete!

  To launch:
     streamlit run app.py

  Then open: http://localhost:8501

  Notes:
  - First run fetches live data from Yahoo Finance
  - Data is cached locally after first fetch
  - LSTM training takes ~1-2 min per asset
{'='*55}
""")
