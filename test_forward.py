#!/usr/bin/env python3
"""
test_forward.py — verify feedforward tinycmax pipeline.
Run from project root:  python test_forward.py
"""
import sys, types, itertools, pathlib
sys.path.insert(0, ".")

# ── Stub cuda_event_ops before any import loads the C extension ──────────────
# The compiled .so files are Python 3.12; this environment is 3.10.
# The linear-warp path (accumulation_window=1) only needs trilinear_splat from
# iwe.py.  We replace it with a pure-PyTorch version that keeps grad_fn intact.

import torch


def _pt_trilinear_splat(events, grid_resolution):
    """
    Pure-PyTorch drop-in for trilinear_splat_cuda.

    events          : [b, n, 4]  (x, y, t, val) float32
    grid_resolution : (d, h, w)  — splat into d+1 time bins
    Returns         : [b, d+1, h, w]

    Gradients propagate through the bilinear weights wx, wy, wt back to
    warped event positions and therefore to the network's flow output.
    """
    b, n, _ = events.shape
    d, h, w = grid_resolution
    x, y, t, val = events.unbind(-1)   # [b, n] each

    parts = []
    for x_off, y_off, t_off in itertools.product((0, 1), (0, 1), (0, 1)):
        # Integer corner indices — detached, no gradient through position
        xi = (x.detach().floor() + x_off).long()
        yi = (y.detach().floor() + y_off).long()
        ti = (t.detach().floor() + t_off).long()

        # Trilinear weights — differentiable w.r.t. x, y, t (and thus flow)
        wx = (x - x.detach().floor()) if x_off else (1.0 - x + x.detach().floor())
        wy = (y - y.detach().floor()) if y_off else (1.0 - y + y.detach().floor())
        wt = (t - t.detach().floor()) if t_off else (1.0 - t + t.detach().floor())

        valid = (
            (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (ti >= 0) & (ti <= d)
        ).float()
        contrib = wx * wy * wt * val * valid   # [b, n], keeps grad_fn

        xi = xi.clamp(0, w - 1)
        yi = yi.clamp(0, h - 1)
        ti = ti.clamp(0, d)
        flat = ti * h * w + yi * w + xi        # [b, n], integer indices

        # scatter_add (non-in-place) returns a new tensor with grad_fn
        buf = torch.zeros(b, (d + 1) * h * w, dtype=events.dtype, device=events.device)
        parts.append(buf.scatter_add(1, flat, contrib))

    return sum(parts).view(b, d + 1, h, w)


# Register stubs before anything tries to import cuda_event_ops
_cuda_stub = types.ModuleType("cuda_event_ops.cuda")
_cuda_stub.trilinear_splat = _pt_trilinear_splat
_cuda_stub.iterative_3d_warp = None   # not used by the linear warp path

_pkg_stub = types.ModuleType("cuda_event_ops")
sys.modules["cuda_event_ops"]        = _pkg_stub
sys.modules["cuda_event_ops.cuda"]   = _cuda_stub

# Load torch_batch (pure PyTorch) directly from source, bypassing __init__
def _load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_ops_dir = pathlib.Path(__file__).parent / "cuda_event_ops" / "cuda_event_ops"
_tb = _load_module("cuda_event_ops.torch_batch", _ops_dir / "torch_batch.py")

# ── Imports ──────────────────────────────────────────────────────────────────
import torch.nn as nn
from dotmap import DotMap
from tinycmax.network import FlowNetwork
from tinycmax.cmax   import ContrastMaximization

# ── 1. Network forward pass ──────────────────────────────────────────────────
print("=" * 56)
print("1. FlowNetwork — forward pass")
print("=" * 56)

net = FlowNetwork(
    encoder_channels=64,
    memory_channels=64,
    decoder_channels=64,
    activation_fn=nn.ReLU,
    final_bias=True,
    padding_mode="reflect",
    scaling=32,
)
net.eval()

B, H, W = 2, 192, 344
x = torch.randn(B, 10, H, W)
with torch.no_grad():
    out = net(DotMap(events=x, _dynamic=False))

flow = out["flow"]
print(f"  input  : {list(x.shape)}")
print(f"  output : {list(flow.shape)}   <- expected [2, 2, 192, 344]")
assert flow.shape == (B, 2, H, W), f"wrong shape: {flow.shape}"
print("  PASS\n")

# ── 2. CMax loss — forward + backward ────────────────────────────────────────
print("=" * 56)
print("2. ContrastMaximization  (accumulation_window=1, base=1, linear)")
print("=" * 56)

loss_fn = ContrastMaximization(accumulation_window=1, base=1, warp="linear")

# Re-run with grad enabled so backward reaches network parameters
net.train()
x2   = torch.randn(B, 10, H, W)
pred = net(DotMap(events=x2, _dynamic=False))

# Synthetic events: [B, N, 4] columns = (t, y, x, p)
#   t : global timestamp in [0, 1] across the N=5 stacked windows
#   y : pixel row  in [0, H-1]
#   x : pixel col  in [0, W-1]
#   p : polarity   in {-1, +1}
N = 500
torch.manual_seed(42)
events = torch.stack([
    torch.rand(B, N),                               # t
    torch.rand(B, N) * (H - 1),                    # y
    torch.rand(B, N) * (W - 1),                    # x
    (torch.randint(0, 2, (B, N)) * 2 - 1).float(), # p
], dim=-1)                                          # [B, N, 4]
counts = torch.full((B,), N)                        # [B]

aux    = DotMap(events=events, counts=counts, _dynamic=False)
target = DotMap(_dynamic=False)

# --- forward (accumulates buffer, increments passes) ---
loss_fn(pred, aux, target)
print(f"  passes after forward : {loss_fn.passes}  "
      f"(== accumulation_window {loss_fn.accumulation_window}  → backward fires)")

# --- backward (computes and returns total_loss tensor) ---
loss_val = loss_fn.backward()
assert loss_val is not None,           "backward returned None (no events?)"
assert not torch.isnan(loss_val),      "loss is NaN"
assert not torch.isinf(loss_val),      "loss is Inf"
print(f"  loss value           : {loss_val.item():.6f}")

# --- propagate to network parameters ---
loss_val.backward()

grads   = [p.grad for p in net.parameters() if p.grad is not None]
max_g   = max(g.abs().max().item() for g in grads) if grads else 0.0
print(f"  max |grad| over all params : {max_g:.4e}")
assert max_g > 0, "no gradient reached any network parameter"
print("  PASS\n")

# ── Summary ──────────────────────────────────────────────────────────────────
print("=" * 56)
print("Summary")
print("=" * 56)
print(f"  flow output shape : {list(flow.shape)}")
print(f"  CMax loss value   : {loss_val.item():.6f}")
