from pathlib import Path
import os, sys

# Fixes torchcodec and ffmpeg issues
ffmpeg_dll_dir = Path(r"C:/Users/jackm/miniconda3/Library/bin")  # adjust if your conda root differs
assert ffmpeg_dll_dir.exists(), ffmpeg_dll_dir
os.add_dll_directory(str(ffmpeg_dll_dir))  # Python 3.8+ DLL search

import t, platform, subprocess
import torchcodec
print("exe:", sys.executable)
print("torch", torch.__version__, "torchcodec", torchcodec.__version__, "py", platform.python_version())
subprocess.run(["ffmpeg", "-version"], check=True)