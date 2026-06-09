#!/usr/bin/env python3
"""
Convert tinycmax optical flow model: PyTorch state_dict → ONNX → TF SavedModel → TFLite

Architecture notes:
  - FlowNetwork: feedforward, no recurrent state
  - Input: [B, 2*N, H, W] — N=5 stacked 2-channel event frames (10 channels total)
  - Output: [B, 2, H, W] — optical flow (u, v)
  - LazyConv2d in encoder: needs one forward pass before loading weights
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from dotmap import DotMap

from tinycmax.network import FlowNetwork

# ── paths ──────────────────────────────────────────────────────────────────
WEIGHTS_PATH = Path("state_dict.pt")
ONNX_PATH    = PROJECT_ROOT / "tinycmax.onnx"
TF_DIR            = PROJECT_ROOT / "tinycmax_tf"
TFLITE_PATH       = PROJECT_ROOT / "tinycmax_ff.tflite"
TFLITE_QUANT_PATH = PROJECT_ROOT / "tinycmax_ff_int8.tflite"

INPUT_SHAPE  = (1, 10, 192, 344)   # batch=1, 2*N=10 stacked channels, MVSEC resolution
OUTPUT_SHAPE = (1, 2, 24, 43)   # 1/8 resolution, upsample done on-board    # batch=1, flow (u,v), same spatial dims
OPSET        = 13


# ── ONNX export wrapper ────────────────────────────────────────────────────
class ExportWrapper(nn.Module):
    """
    Thin wrapper that accepts a plain events tensor instead of a DotMap
    and returns only the flow tensor, for clean ONNX tracing.
    """

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        out = FlowNetwork.forward(self.network, {"events": events})
        return out["flow"]


# ── step 1: load model ─────────────────────────────────────────────────────
def load_network() -> nn.Module:
    print("─" * 55)
    print("Step 1: Load network")
    print("─" * 55)

    print("  Instantiating FlowNetwork (feedforward, 10-channel input) ...")
    net = FlowNetwork(
        encoder_channels=64,
        memory_channels=64,
        decoder_channels=64,
        activation_fn=nn.ReLU,
        final_bias=True,
        padding_mode="zeros",   # changed from "reflect"
        scaling=32,
    )

    # LazyConv2d in the encoder has no weight until the first forward pass
    print("  Initialising lazy layers with dummy forward pass ...")
    net.eval()
    dummy_dm = DotMap({"events": torch.zeros(*INPUT_SHAPE)}, _dynamic=False)
    with torch.no_grad():
        net(dummy_dm)

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
    # Remove bilinear upsample - not supported on N6, will do in MicroPython
    net.decoder.dec2 = torch.nn.Identity()

    wrapper = ExportWrapper(net)
    
    wrapper.eval()

    dummy = torch.zeros(*INPUT_SHAPE)

    # sanity-check the wrapper before tracing
    with torch.no_grad():
        ref = wrapper(dummy)
    print(f"  Wrapper reference output shape : {tuple(ref.shape)}")
    assert tuple(ref.shape) == OUTPUT_SHAPE, f"Expected {OUTPUT_SHAPE}, got {ref.shape}"

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


# ── step 5a: TF SavedModel → TFLite (float32) ────────────────────────────
def saved_model_to_tflite(tf_dir: Path) -> Path:
    print()
    print("─" * 55)
    print("Step 5a: TF SavedModel → TFLite  (float32)")
    print("─" * 55)

    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    tflite_model = converter.convert()

    TFLITE_PATH.write_bytes(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"  TFLite model saved : {TFLITE_PATH} ({size_mb:.2f} MB)")
    return TFLITE_PATH


# ── step 5b: TF SavedModel → TFLite (INT8 PTQ, float IO) ─────────────────
def saved_model_to_tflite_int8(tf_dir: Path) -> Path:
    print()
    print("─" * 55)
    print("Step 5b: TF SavedModel → TFLite  (INT8 PTQ, float IO)")
    print("─" * 55)

    import tensorflow as tf

    def representative_dataset():
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from tinycmax.mvsec import MvsecSequence
        from torch.utils.data import DataLoader

        dataset = MvsecSequence(
            root_dir="/content/tinycmax/data/mvsec",
            recording="outdoor_day1",
            time_window=0.05,
            count_window=None,
            n_stack=5,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        count = 0
        for batch in loader:
            if count >= 200:
                break
            events = batch["events"].numpy()  # (1, 10, 192, 344) NCHW
            events_nhwc = events.transpose(0, 2, 3, 1).astype(np.float32)  # (1, 192, 344, 10)
            yield [events_nhwc]
            count += 1
        print(f"  Calibration: used {count} real MVSEC samples")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # float IO: inputs and outputs stay float32; only internal ops are quantized
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    TFLITE_QUANT_PATH.write_bytes(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"  TFLite INT8 model saved : {TFLITE_QUANT_PATH} ({size_mb:.2f} MB)")
    return TFLITE_QUANT_PATH


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

    # Check spatial dims and channel count match the flow output spec
    n, c, h, w = OUTPUT_SHAPE
    expected = [n, h, w, c]
    if list(result.shape) == expected:
        print(f"  Output shape correct (NHWC {expected} == PyTorch NCHW [{n}, {c}, {h}, {w}]).")
    else:
        print(f"  WARNING: output shape {list(result.shape)} differs from expected {expected} (NHWC).")


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    net             = load_network()
    onnx_path       = export_onnx(net)
    onnx_path       = simplify_onnx(onnx_path)
    tf_dir          = onnx_to_saved_model(onnx_path)

    tflite_path     = saved_model_to_tflite(tf_dir)
    verify_tflite(tflite_path)

    tflite_int8_path = saved_model_to_tflite_int8(tf_dir)
    verify_tflite(tflite_int8_path)

    print()
    print("=" * 55)
    print("Conversion complete.")
    print(f"  ONNX        : {ONNX_PATH}")
    print(f"  TFLite fp32 : {TFLITE_PATH}")
    print(f"  TFLite int8 : {TFLITE_QUANT_PATH}")
    print("=" * 55)
