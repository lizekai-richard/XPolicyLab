#!/usr/bin/env bash
# =============================================================================
# download_weights.sh -- fetch everything the RoboDojo SANA policy server loads
# at start-up into the SHARED home (run on ONE node, e.g. l5; all four see it):
#   1. policy checkpoint root   <- private HF repo $CKPT_REPO (needs ~/.hf_token)
#        -> ~/checkpoints/huggingface/<repo>   (mirrors the cluster layout)
#   2. LTX-2.3 video VAE        <- Efficient-Large-Model/LTX-2.3-Diffusers, vae/ only
#        -> $SANA_REPO/output/pretrained_models/LTX-2.3-Diffusers  (launcher default)
#   3. gemma-2-2b-it text enc.  <- Efficient-Large-Model/gemma-2-2b-it
#        -> $SANA_REPO/output/pretrained_models/text_encoder/gemma-2-2b-it
#           (what diffusion/model/builder.py maps "gemma-2-2b-it-local" to)
# No null-embed artifact: robodojo/server/main.py serves with cfg_scale=1.0.
# Re-runnable: snapshot_download skips files that are already complete.
# Another checkpoint:  CKPT_REPO=logits/<repo> bash download_weights.sh   (VAE/text encoder are
# skipped automatically once present).  The normalization receipt names a training-cluster input
# cache; if the snapshot does not ship it, a byte-identical copy is taken from a sibling snapshot's
# json_cache/ (same corpus -> same file); if none matches the download FAILS (rc 2): the server's
# normalization gate would refuse to start.
# =============================================================================
set -Eeuo pipefail
ROOT=${ROBODOJO_ROOT:-/dataset2/jiamingt/robodojo}
SANA_REPO=${SANA_REPO:-$HOME/Sana}
CKPT_REPO=${CKPT_REPO:-logits/sana_rwm_policy_from_pretrained_robodojo_320px_epoch4_step25000}
CKPT_ROOT=${CKPT_ROOT:-$HOME/checkpoints/huggingface/$CKPT_REPO}
SKIP_SHARED=${SKIP_SHARED:-auto}   # 1 = policy checkpoint only; auto = skip VAE/text encoder when both are already complete
VAE_REPO=Efficient-Large-Model/LTX-2.3-Diffusers
VAE_DIR=$SANA_REPO/output/pretrained_models/LTX-2.3-Diffusers
TE_REPO=Efficient-Large-Model/gemma-2-2b-it
TE_DIR=$SANA_REPO/output/pretrained_models/text_encoder/gemma-2-2b-it
MIRROR=$HOME/robodojo_install/logs
mkdir -p "$MIRROR" "$(dirname "$CKPT_ROOT")" "$(dirname "$VAE_DIR")" "$(dirname "$TE_DIR")"
exec > >(tee -a "$MIRROR/download_weights.log") 2>&1
log() { printf '[%s] [%s] %s\n' "$(date '+%F %T')" "$(hostname -s)" "$*"; }

[[ -f $ROOT/activate_sana_wam.sh ]] || { log "FATAL: $ROOT/activate_sana_wam.sh missing (sana-wam not installed on this node?)"; exit 1; }
# shellcheck disable=SC1091
source "$ROOT/activate_sana_wam.sh"
[[ -s ${HF_TOKEN_PATH:-/nonexistent} ]] || { log "FATAL: HF token file ${HF_TOKEN_PATH:-unset} missing or empty"; exit 1; }
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
log "python=$SANA_PY  token_file=$HF_TOKEN_PATH ($(stat -c %a "$HF_TOKEN_PATH"))"
log "ckpt: $CKPT_REPO -> $CKPT_ROOT"; log "vae : $VAE_REPO (vae/ only) -> $VAE_DIR"; log "te  : $TE_REPO -> $TE_DIR"

"$SANA_PY" - "$CKPT_REPO" "$CKPT_ROOT" "$VAE_REPO" "$VAE_DIR" "$TE_REPO" "$TE_DIR" "$SKIP_SHARED" <<'PY'
import glob, hashlib, json, os, shutil, sys, time
from huggingface_hub import HfApi, snapshot_download
ckpt_repo, ckpt_root, vae_repo, vae_dir, te_repo, te_dir, skip_shared = sys.argv[1:8]
api = HfApi()
print("token user:", api.whoami().get("name"), flush=True)

def fetch(repo, dest, patterns=None):
    info = api.model_info(repo, files_metadata=True)
    sel = [s for s in info.siblings if patterns is None or any(
        __import__("fnmatch").fnmatch(s.rfilename, p) for p in patterns)]
    tot = sum((s.size or 0) for s in sel)
    print(f"--> {repo}: {len(sel)} files, {tot/1e9:.2f} GB -> {dest}", flush=True)
    t0 = time.time()
    snapshot_download(repo, local_dir=dest, allow_patterns=patterns, max_workers=8)
    dt = time.time() - t0
    print(f"    done in {dt:.0f}s ({tot/1e9/max(dt,1):.2f} GB/s)", flush=True)
    return sel

fetch(ckpt_repo, ckpt_root)
shared_present = all(os.path.exists(p) for p in (
    os.path.join(vae_dir, "vae", "diffusion_pytorch_model.safetensors"), os.path.join(vae_dir, "vae", "config.json"),
    os.path.join(te_dir, "model.safetensors.index.json"), os.path.join(te_dir, "tokenizer.json")))
if skip_shared == "1" or (skip_shared == "auto" and shared_present):
    print(f"--> VAE / text encoder: skipped (SKIP_SHARED={skip_shared}, present={shared_present})", flush=True)
else:
    fetch(vae_repo, vae_dir, ["vae/*", "model_index.json"])
    fetch(te_repo, te_dir)

def need(path, what):
    ok = os.path.exists(path)
    print(("OK  " if ok else "MISSING ") + what + ": " + path, flush=True)
    return ok

ok = True
ok &= need(os.path.join(ckpt_root, "config.yaml"), "policy config")
ok &= need(os.path.join(ckpt_root, "model", "pytorch_model_fsdp.bin"), "policy weights")
ok &= need(os.path.join(ckpt_root, "README.md"), "policy training log")
norms = sorted(glob.glob(os.path.join(ckpt_root, "normalization", "*.json")))
print(("OK  " if norms else "MISSING ") + "normalization json(s): " + ", ".join(os.path.basename(n) for n in norms), flush=True)
ok &= bool(norms)
ok &= need(os.path.join(vae_dir, "vae", "config.json"), "vae config")
ok &= need(os.path.join(vae_dir, "vae", "diffusion_pytorch_model.safetensors"), "vae weights")
ok &= need(os.path.join(te_dir, "config.json"), "text encoder config")
ok &= need(os.path.join(te_dir, "model.safetensors.index.json"), "text encoder shard index")
ok &= need(os.path.join(te_dir, "tokenizer.json"), "tokenizer")

def sha256_hex(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

# The receipt's input cache must be present locally with the recorded sha256 (server gate).
snapshots_dir = os.path.dirname(ckpt_root.rstrip("/"))
for norm in norms:
    try:
        receipt = json.load(open(norm))
    except Exception as e:  # noqa: BLE001
        print("WARN cannot parse", norm, e, flush=True); continue
    for entry in receipt.get("input_caches") or []:
        name = os.path.basename(str(entry.get("path", "")))
        want = str(entry.get("sha256", "")).split(":")[-1].lower()
        if not name or not want:
            print("WARN input cache entry without path/sha256 in", os.path.basename(norm), flush=True); ok = False; continue
        found = None
        for d in (os.path.join(ckpt_root, "json_cache"), os.path.dirname(norm), ckpt_root):
            c = os.path.join(d, name)
            if os.path.isfile(c) and sha256_hex(c) == want:
                found = c; break
        if found is None:
            siblings = glob.glob(os.path.join(snapshots_dir, "*", "json_cache", name)) + glob.glob(os.path.join(snapshots_dir, "*", name))
            for c in siblings:
                if os.path.realpath(c).startswith(os.path.realpath(ckpt_root) + os.sep):
                    continue
                if os.path.isfile(c) and sha256_hex(c) == want:
                    dst_dir = os.path.join(ckpt_root, "json_cache"); os.makedirs(dst_dir, exist_ok=True)
                    dst = os.path.join(dst_dir, name); shutil.copy2(c, dst); found = dst
                    print("OK  normalization input cache copied from sibling snapshot:", c, "->", dst, flush=True)
                    break
        if found:
            print("OK  normalization input cache:", found, "sha256", want[:12] + "...", flush=True)
        else:
            print("MISSING normalization input cache", name, "sha256", want[:12] + "... : not in this snapshot and no sibling "
                  "snapshot has a byte-identical copy -> add json_cache/" + name + " to the HF repo (or copy it into "
                  + os.path.join(ckpt_root, "json_cache") + ")", flush=True)
            ok = False

try:
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ckpt_root, "config.yaml")))
    def g(*ks):
        d = cfg
        for k in ks:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d if d != {} else None
    print("config.yaml: model=", g("model", "model"), "image_size=", g("model", "image_size"),
          "| text_encoder=", g("text_encoder", "text_encoder_name"), "| vae=", g("vae", "vae_type"), g("vae", "vae_pretrained"),
          "| inference_cfg_scale=", g("scheduler", "inference_cfg_scale"), "| null_embed_root=", g("train", "null_embed_root"), flush=True)
    ds = g("data", "data_dir") or g("data", "datasets") or g("data", "type")
    print("config.yaml: data=", str(ds)[:200], flush=True)
except Exception as e:  # noqa: BLE001
    print("config.yaml summary failed:", e, flush=True)

def du(p):
    t = 0
    for r, _, fs in os.walk(p):
        if "/.cache" in r: continue
        for f in fs:
            try: t += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return t/1e9
print(f"sizes: ckpt {du(ckpt_root):.2f} GB | vae {du(vae_dir):.2f} GB | text_encoder {du(te_dir):.2f} GB", flush=True)
sys.exit(0 if ok else 2)
PY
rc=$?
if (( rc == 0 )); then
  { echo "date=$(date -Is)"; echo "ckpt_repo=$CKPT_REPO"; echo "ckpt_root=$CKPT_ROOT"; echo "vae_dir=$VAE_DIR"; echo "text_encoder_dir=$TE_DIR"; } > "$MIRROR/download_weights.DONE"
  cp "$MIRROR/download_weights.DONE" "$MIRROR/download_weights.$(basename "$CKPT_REPO").DONE"
  log "ALL WEIGHTS DOWNLOADED AND VERIFIED"
else
  log "download/verification FAILED (rc=$rc); re-run this script to resume"
fi
exit $rc
