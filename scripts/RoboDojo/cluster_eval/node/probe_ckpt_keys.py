#!/usr/bin/env python
"""List the state-dict keys of torch zip checkpoints WITHOUT downloading the weights.

Each source is either a local path or an HF path ``hf://<owner>/<repo>/<file>``; HF files are opened through
``HfFileSystem`` so only the zip central directory and ``data.pkl`` are fetched (HTTP range reads, well under 1 MB).
The pickle is replayed with stub classes, so no tensor storage is ever materialized.

Prints, per source: file size, key count, total numel (params + buffers), the MoT text-path layout as the SANA_MOT
adapter classifies it (``detect_context_layout``), the text-path keys with shapes, and a digest of the sorted
(key, shape) list so two checkpoints can be compared for layout identity.

Usage (node, sana-wam env): python probe_ckpt_keys.py [--dump DIR] hf://logits/<repo>/model/pytorch_model_fsdp.bin /path/to/local.bin
(--dump writes DIR/<source>.keys.json with every key's shape, for an offline strict-load rehearsal against a mirror.)
"""

from __future__ import annotations

import collections
import hashlib
import io
import os
import pickle
import sys
import zipfile


class _Stub:
    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


def _rebuild_tensor(storage, offset, size, stride, *rest):
    return ("tensor", tuple(size), storage[1] if isinstance(storage, tuple) else "?")


def _rebuild_parameter(data, requires_grad, backward_hooks, *rest):
    return data


def _rebuild_from_type_v2(func, new_type, args, state):
    return func(*args)


class _KeyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name in ("_rebuild_parameter", "_rebuild_parameter_with_state"):
            return _rebuild_parameter
        if module == "torch._tensor" and name == "_rebuild_from_type_v2":
            return _rebuild_from_type_v2
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "torch" and name in ("float32", "bfloat16", "float16", "float64", "int64", "int32", "int16", "bool", "uint8"):
            return name
        return _Stub

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        return ("storage", str(pid[1]), pid[2], pid[3], pid[4])


def _open(source: str):
    if source.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        token = None
        token_path = os.environ.get("HF_TOKEN_PATH")
        if token_path and os.path.isfile(token_path):
            token = open(token_path).read().strip()
        fs = HfFileSystem(token=token)
        path = source[len("hf://"):]
        size = fs.info(path)["size"]
        return fs.open(path, "rb", block_size=1 << 20), size
    return open(source, "rb"), os.path.getsize(source)


def _layout(keys) -> str:
    if "action_dit.context_mlp.0.weight" in keys:
        return "legacy_action_mlp"
    if "context_embedder.action_y_embedder.y_proj.fc1.weight" in keys:
        return "context_embedder"
    if any(k.startswith("context_embedder.") for k in keys):
        return "TRANSIENT (refused by the adapter)"
    return "UNKNOWN (not a MoT policy state dict)"


def probe(source: str, dump_dir: str | None = None) -> None:
    handle, size = _open(source)
    with handle, zipfile.ZipFile(handle) as archive:
        pkl_name = next(n for n in archive.namelist() if n.endswith("/data.pkl") or n == "data.pkl")
        raw = archive.read(pkl_name)
        entries = len(archive.namelist())
    obj = _KeyOnlyUnpickler(io.BytesIO(raw)).load()
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    rows = {k.removeprefix("module."): v for k, v in state.items()}
    shapes = {k: (v[1] if isinstance(v, tuple) and v and v[0] == "tensor" else None) for k, v in rows.items()}
    numel = 0
    for shape in shapes.values():
        if shape is not None:
            n = 1
            for d in shape:
                n *= d
            numel += n
    digest = hashlib.sha256("\n".join(f"{k} {shapes[k]}" for k in sorted(shapes)).encode()).hexdigest()[:16]
    blocks = sorted({int(k.split(".")[1]) for k in rows if k.startswith("blocks.")})
    text_keys = sorted(
        k
        for k in rows
        if k.startswith("context_embedder.")
        or ".y_embedder." in k
        or "y_norm" in k
        or "context_mlp" in k
        or "state_" in k
        or k == "video_dit.pos_embed"
    )
    print(f"== {source}")
    print(f"   file bytes {size:,}  zip entries {entries}  data.pkl {len(raw):,} B  keys {len(rows)}  numel {numel:,}")
    print(f"   blocks {len(blocks)} ({blocks[:1]}..{blocks[-1:]})  text layout: {_layout(rows)}  key/shape digest {digest}")
    for k in text_keys:
        print(f"     {k:60s} {shapes[k]}")
    missing = sorted(k for k, v in shapes.items() if v is None)
    print(f"   entries without a recovered shape: {len(missing)} {missing[:3]}")
    # storage type per tensor (FloatStorage, BFloat16Storage, ...): the minority types are listed key by key
    dtypes = {k: (v[2] if isinstance(v, tuple) and len(v) > 2 else None) for k, v in rows.items()}
    counts = collections.Counter(dtypes.values())
    print(f"   storage types: {dict(counts)}")
    majority = counts.most_common(1)[0][0] if counts else None
    for k in sorted(k for k, t in dtypes.items() if t != majority):
        print(f"     {dtypes[k]:18s} {k} {shapes[k]}")
    if dump_dir:
        import json

        os.makedirs(dump_dir, exist_ok=True)
        label = source.removeprefix("hf://").replace("/", "__")
        with open(os.path.join(dump_dir, label + ".keys.json"), "w") as handle:
            json.dump(
                {
                    "source": source,
                    "file_bytes": size,
                    "shapes": {k: shapes[k] for k in sorted(shapes)},
                    "dtypes": {k: dtypes[k] for k in sorted(dtypes)},
                },
                handle,
            )
    sys.stdout.flush()


if __name__ == "__main__":
    argv = sys.argv[1:]
    dump = None
    if argv[:1] == ["--dump"]:
        dump, argv = argv[1], argv[2:]
    for arg in argv:
        try:
            probe(arg, dump)
        except Exception as exc:  # keep going over the remaining sources
            print(f"== {arg}\n   FAILED: {type(exc).__name__}: {exc}")
