#!/usr/bin/env python3
"""
Convert tinycmax optical flow model: PyTorch state_dict → ONNX → TF SavedModel → TFLite

Architecture notes:
  - WrappedFlowNetwork(NetworkWrapper, FlowNetwork)
  - NetworkWrapper.forward manages GRU hidden state in self.state
  - FlowNetwork.forward takes {"events": tensor} + explicit hidden, returns ({"flow": tensor}, hidden)
  - All Conv2d layers are LazyConv2d (need one forward pass before loading weights)
  - For ONNX export: call FlowNetwork.forward directly with hidden=None (stateless, GRU inits to zeros)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from hydra.utils import instantiate
from dotmap import DotMap

from tinycmax.network import FlowNetwork

# ── paths ──────────────────────────────────────────────────────────────────
CKPT_DIR     = PROJECT_ROOT / "checkpoints" / "rb5gp4fv_latest"
CONFIG_PATH  = CKPT_DIR / "config.yaml"
WEIGHTS_PATH = CKPT_DIR / "state_dict.pt"
ONNX_PATH    = PROJECT_ROOT / "tinycmax.onnx"
TF_DIR       = PROJECT_ROOT / "tinycmax_tf"
TFLITE_PATH  = PROJECT_ROOT / "tinycmax_tid.tflite"

INPUT_SHAPE = (1, 2, 192, 344)
OPSET       = 13


# ── ONNX export wrapper ────────────────────────────────────────────────────
class ExportWrapper(nn.Module):
    """
    Thin wrapper that:
    - accepts a plain events tensor instead of a DotMap
    - calls FlowNetwork.forward directly (bypasses NetworkWrapper's stateful forward)
    - passes hidden=None so the GRU initialises to zeros each call (stateless export)
    - returns only the flow tensor
    """

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        out, _ = FlowNetwork.forward(self.network, {"events": events}, None)
        return out["flow"]


# ── step 1: load model ─────────────────────────────────────────────────────
def load_network() -> nn.Module:
    print("─" * 55)
    print("Step 1: Load network")
    print("─" * 55)

    cfg = OmegaConf.load(CONFIG_PATH)

    print("  Instantiating WrappedFlowNetwork from config ...")
    net = instantiate(cfg.network)

    # LazyConv2d layers have no weight until the first forward pass
    print("  Initialising lazy layers with dummy forward pass ...")
    net.eval()
    dummy_dm = DotMap({"events": torch.zeros(*INPUT_SHAPE)}, _dynamic=False)
    with torch.no_grad():
        net(dummy_dm)
    net.reset_state()

    print("  Loading state dict ...")
    sd = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    net.load_state_dict(sd)
    net.eval()

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Parameters: {n_params:,}")
    return net


# ── step 2: export ONNX ────────────────────────────────────────────────────
def export_onnx(net: nn.Module) -> Path:
    print()
    print("─" * 55)
    print("Step 2: Export to ONNX")
    print("─" * 55)

    wrapper = ExportWrapper(net)
    wrapper.eval()

    dummy = torch.zeros(*INPUT_SHAPE)

    # sanity-check the wrapper before tracing
    with torch.no_grad():
        ref = wrapper(dummy)
    print(f"  Wrapper reference output shape : {tuple(ref.shape)}")
    assert tuple(ref.shape) == INPUT_SHAPE, f"Expected {INPUT_SHAPE}, got {ref.shape}"

    print(f"  Tracing and exporting (opset {OPSET}) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(ONNX_PATH),
            input_names=["events"],
            output_names=["flow"],
            opset_version=OPSET,
            do_constant_folding=True,
        )

    import onnx
    model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model)
    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"  ONNX model saved  : {ONNX_PATH} ({size_mb:.2f} MB, valid)")
    return ONNX_PATH


# ── step 3: simplify ONNX ─────────────────────────────────────────────────
def simplify_onnx(onnx_path: Path) -> Path:
    print()
    print("─" * 55)
    print("Step 3: Simplify ONNX (onnxsim)")
    print("─" * 55)
    try:
        import onnx
        from onnxsim import simplify
        model = onnx.load(str(onnx_path))
        model_sim, ok = simplify(model)
        if ok:
            onnx.save(model_sim, str(onnx_path))
            print("  Simplification succeeded, model overwritten.")
        else:
            print("  Simplification check failed, keeping original.")
    except Exception as e:
        print(f"  onnxsim skipped ({e}), continuing with unsimplified model.")
    return onnx_path


# ── step 4: ONNX → TF SavedModel ──────────────────────────────────────────
def onnx_to_saved_model(onnx_path: Path) -> Path:
    print()
    print("─" * 55)
    print("Step 4: ONNX → TF SavedModel  (onnx2tf)")
    print("─" * 55)

    import onnx2tf
    TF_DIR.mkdir(parents=True, exist_ok=True)

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(TF_DIR),
        not_use_onnxsim=True,   # already simplified above
        disable_group_convolution=True,
    )
    print(f"  TF SavedModel saved : {TF_DIR}")
    return TF_DIR


# ── step 5: TF SavedModel → TFLite ────────────────────────────────────────
def saved_model_to_tflite(tf_dir: Path) -> Path:
    print()
    print("─" * 55)
    print("Step 5: TF SavedModel → TFLite")
    print("─" * 55)

    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    tflite_model = converter.convert()

    TFLITE_PATH.write_bytes(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"  TFLite model saved : {TFLITE_PATH} ({size_mb:.2f} MB)")
    return TFLITE_PATH


# ── step 6: verify TFLite ─────────────────────────────────────────────────
def verify_tflite(tflite_path: Path) -> None:
    print()
    print("─" * 55)
    print("Step 6: Verify TFLite model")
    print("─" * 55)

    # ai-edge-litert ships its own Interpreter (preferred over tf.lite.Interpreter)
    try:
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=str(tflite_path))
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(tflite_path))

    interp.allocate_tensors()
    inp_detail  = interp.get_input_details()[0]
    out_detail  = interp.get_output_details()[0]

    # onnx2tf converts to NHWC (channel-last), so TFLite I/O shape is [N, H, W, C]
    # rather than PyTorch's NCHW [N, C, H, W].  Use the model's own reported shapes.
    tflite_in_shape  = inp_detail["shape"].tolist()
    tflite_out_shape = out_detail["shape"].tolist()

    print(f"  Input  : name={inp_detail['name']!r:40s} shape={tflite_in_shape}  dtype={inp_detail['dtype'].__name__}")
    print(f"  Output : name={out_detail['name']!r:40s} shape={tflite_out_shape}  dtype={out_detail['dtype'].__name__}")
    print(f"  NOTE: TFLite uses NHWC layout — feed/read tensors as [N, H, W, C], not PyTorch's [N, C, H, W]")

    dummy_np = np.zeros(tflite_in_shape, dtype=np.float32)
    interp.set_tensor(inp_detail["index"], dummy_np)
    interp.invoke()
    result = interp.get_tensor(out_detail["index"])

    print(f"  Inference output shape: {list(result.shape)}")

    # Verify spatial dims and channel count match the original PyTorch spec
    n, c, h, w = INPUT_SHAPE
    assert list(result.shape) == [n, h, w, c], (
        f"Unexpected output shape: got {list(result.shape)}, expected [{n}, {h}, {w}, {c}] (NHWC)"
    )
    print(f"  Output shape correct (NHWC [{n}, {h}, {w}, {c}] == PyTorch NCHW [{n}, {c}, {h}, {w}]).")


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    net        = load_network()
    onnx_path  = export_onnx(net)
    onnx_path  = simplify_onnx(onnx_path)
    tf_dir     = onnx_to_saved_model(onnx_path)
    tflite_path = saved_model_to_tflite(tf_dir)
    verify_tflite(tflite_path)

    print()
    print("=" * 55)
    print(f"Conversion complete.")
    print(f"  ONNX   : {ONNX_PATH}")
    print(f"  TFLite : {TFLITE_PATH}")
    print("=" * 55)
