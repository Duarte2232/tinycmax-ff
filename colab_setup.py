# ============================================================
# CELL 1 — Version check
# ============================================================
import subprocess, sys, torch

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()

print("── Python ──────────────────────────────────────────")
print(run("python --version"))

print("\n── PyTorch ─────────────────────────────────────────")
print(f"torch {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute cap    : {cap[0]}.{cap[1]}")

print("\n── nvcc ────────────────────────────────────────────")
print(run("nvcc --version") or "nvcc not found — add /usr/local/cuda/bin to PATH")


# ============================================================
# CELL 2 — Clone repo + fix submodule URL + build CUDA ops
# ============================================================
import os, subprocess, sys

REPO_URL   = "https://github.com/Huizerd/tinycmax.git"
REPO_DIR   = "/content/tinycmax"
# Submodule uses SSH (git@github.com:tudelft/cuda_event_ops.git) — replace
# with HTTPS so Colab doesn't need an SSH key.
SUBMOD_URL = "https://github.com/tudelft/cuda_event_ops.git"

def sh(cmd, cwd=None):
    """Run a shell command, stream output, raise on failure."""
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode})")
    return result.stdout

# ── 1. Clone main repo ───────────────────────────────────────
if not os.path.isdir(REPO_DIR):
    sh(f"git clone {REPO_URL} {REPO_DIR}")
else:
    print(f"{REPO_DIR} already exists, skipping clone")

# ── 2. Override submodule URL (SSH → HTTPS) and initialise ───
sh("git submodule set-url cuda_event_ops " + SUBMOD_URL, cwd=REPO_DIR)
sh("git submodule update --init --recursive", cwd=REPO_DIR)

# ── 3. Detect compute capability and build CUDA extension ────
cap = torch.cuda.get_device_capability(0)
arch = f"{cap[0]}.{cap[1]}"
print(f"\nBuilding for sm_{cap[0]}{cap[1]}  (compute cap {arch})")

CUDA_EXT_DIR = os.path.join(REPO_DIR, "cuda_event_ops")
env = os.environ.copy()
env["TORCH_CUDA_ARCH_LIST"] = arch   # e.g. "7.5" for T4, "8.0" for A100
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "."],
    cwd=CUDA_EXT_DIR,
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(result.stdout[-3000:])  # tail — full output can be long
if result.returncode != 0:
    raise RuntimeError("cuda_event_ops build failed — see output above")
print("Build complete")

# ── 4. Put tinycmax on sys.path ──────────────────────────────
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print(f"\nAdded {REPO_DIR} to sys.path")


# ============================================================
# CELL 3 — Verify trilinear_splat install
# ============================================================
import torch
from cuda_event_ops.cuda import trilinear_splat

print("Import OK:", trilinear_splat)

# Sanity call:
#   input  : [b=1, n=100, 4]  — (x, y, z, val) on CUDA, contiguous
#   grid   : (d=1, h=64, w=64)
#   output : [b, d, h, w] = [1, 1, 64, 64]
events = torch.randn(1, 100, 4).cuda().contiguous()
out    = trilinear_splat(events, (1, 64, 64))

print(f"\nInput  : {list(events.shape)}  (x, y, z, val) — CUDA, contiguous")
print(f"Output : {list(out.shape)}   <- expected [1, 1, 64, 64]")
assert out.shape == (1, 1, 64, 64), f"unexpected shape: {out.shape}"
print("PASS — trilinear_splat_cuda is working")


# ============================================================
# CELL 4 — Mount Google Drive and set MVSEC data path
# ============================================================
from google.colab import drive
drive.mount("/content/drive")

# ── Adjust this to wherever you stored the MVSEC data ────────
# Expected directory layout inside the folder you choose:
#
#   mvsec/
#     outdoor_day/
#       outdoor_day1_data.hdf5
#       outdoor_day1_gt.hdf5
#       outdoor_day2_data.hdf5
#       outdoor_day2_gt.hdf5
#       calib/
#         ...
#     indoor_flying/
#       ...
#
DRIVE_ROOT   = "/content/drive/MyDrive"
MVSEC_DIR    = f"{DRIVE_ROOT}/datasets/mvsec"   # ← edit this

import os
assert os.path.isdir(MVSEC_DIR), (
    f"MVSEC directory not found: {MVSEC_DIR}\n"
    "Upload the data to Google Drive and update MVSEC_DIR above."
)

print(f"MVSEC root : {MVSEC_DIR}")
print("Contents   :", os.listdir(MVSEC_DIR))

# Optionally symlink into the repo's expected data/ location
SYMLINK = "/content/tinycmax/data/mvsec"
os.makedirs(os.path.dirname(SYMLINK), exist_ok=True)
if not os.path.lexists(SYMLINK):
    os.symlink(MVSEC_DIR, SYMLINK)
    print(f"Symlinked  : {SYMLINK} → {MVSEC_DIR}")
else:
    print(f"Symlink already exists: {SYMLINK}")
