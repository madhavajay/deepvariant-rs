#!/usr/bin/env python3
"""
Fix WGS model.onnx so the CoreML MLProgram converter accepts it,
*and* runs fast.

Two issues with the raw tf2onnx export:

1. Some Conv/MaxPool nodes have neither `pads` nor `auto_pad` set.
   ONNX defaults that to no-padding, but CoreML MLProgram refuses
   the implicit default and errors with `Required param 'pad' is
   missing`. We add `pads=[0,...]` explicitly.

2. The batch dimension is dynamic (`unk__N`). MLProgram compiles a
   specialized graph and then rejects any inference call whose batch
   shape doesn't match, falling back to the CPU EP for those batches
   (we measured this as a net wall-time regression). Pinning the
   batch dim to 1 keeps the spec static and lets CoreML stay on
   GPU+ANE for every call; the caller pads / runs batch=1
   per-example. Set DV_BATCH_DIM=128 to pin to 128 instead (caller
   must then pad partial batches).
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx


def pin_batch_dim(model: onnx.ModelProto, batch: int) -> int:
    """Replace the symbolic batch dim on graph inputs/outputs with a
    fixed integer. Returns the number of tensors updated."""
    n = 0
    for tensor in list(model.graph.input) + list(model.graph.output):
        shape = tensor.type.tensor_type.shape
        if not shape.dim:
            continue
        dim0 = shape.dim[0]
        if dim0.dim_value == 0:  # symbolic or unset
            dim0.dim_param = ""  # clear any symbol
            dim0.dim_value = batch
            n += 1
    return n


def main(src: Path, dst: Path) -> None:
    # All ONNX ops that take a `pads` attribute and have a sensible
    # default. We add `pads=[0,...]` (no-padding) when neither `pads`
    # nor `auto_pad` is set, matching the ONNX spec default.
    PAD_OPS = {"Conv", "ConvTranspose", "MaxPool", "AveragePool", "LpPool"}

    import os
    batch_dim = int(os.environ.get("DV_BATCH_DIM", "1"))

    model = onnx.load(str(src))
    n_pinned = pin_batch_dim(model, batch_dim)
    print(f"Pinned batch dim to {batch_dim} on {n_pinned} tensor(s)")
    n_fixed = 0
    n_by_op: dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type not in PAD_OPS:
            continue
        attrs = {a.name: a for a in node.attribute}
        if "pads" in attrs or "auto_pad" in attrs:
            continue
        kernel = attrs["kernel_shape"].ints if "kernel_shape" in attrs else None
        if kernel is None:
            print(
                f"WARN: {node.op_type} {node.name!r} missing kernel_shape — skipping",
                file=sys.stderr,
            )
            continue
        rank = len(kernel)
        node.attribute.append(
            onnx.helper.make_attribute("pads", [0] * (2 * rank))
        )
        n_fixed += 1
        n_by_op[node.op_type] = n_by_op.get(node.op_type, 0) + 1
    print(
        f"Added explicit pads=[0,...] to {n_fixed} nodes "
        f"({', '.join(f'{c} {op}' for op, c in sorted(n_by_op.items()))})"
    )
    onnx.save(model, str(dst))
    print(f"Wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <input.onnx> <output.onnx>", file=sys.stderr)
        sys.exit(2)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
