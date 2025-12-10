from pathlib import Path
import os, sys

# Fixes torchcodec and ffmpeg issues
import torch, platform, subprocess
import torchcodec
print("exe:", sys.executable)
print("torch", torch.__version__, "torchcodec", torchcodec.__version__, "py", platform.python_version())
subprocess.run(["ffmpeg", "-version"], check=True)