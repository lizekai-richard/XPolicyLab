#!/usr/bin/env bash
# =============================================================================
# download_hf_repo.sh -- fetch a WHOLE HF model repo into the shared NFS home, as-is.
# For checkpoints whose layout is NOT the SANA work_dir one (download_partial_ckpt.sh handles that):
# e.g. OpenWAM bundles = config.yaml + checkpoint_step_*.safetensors + normalization_stats.npy + tokenizer/.
# Runs on ONE MIT node (shared home). Env:
#   REPO        HF repo id, e.g. logits/openwam_robodojo
#   LABEL       destination name: ~/checkpoints/huggingface/logits/<LABEL>   (default: basename of REPO)
#   ALLOW       optional space-separated allow_patterns (default: everything)
# Verifies every file against the repo's SHA256SUMS when it ships one. Re-runnable (skips complete files).
# =============================================================================
set -Eeuo pipefail
ROOT=${ROBODOJO_ROOT:-/dataset2/jiamingt/robodojo}
: "${REPO:?REPO required}"; LABEL=${LABEL:-$(basename "$REPO")}
DEST=${DEST:-$HOME/checkpoints/huggingface/logits/$LABEL}
LOGS=$HOME/robodojo_install/logs; mkdir -p "$LOGS" "$DEST"
exec > >(tee -a "$LOGS/download_hf_repo.log") 2>&1
log() { printf '[%s] [%s] %s\n' "$(date '+%F %T')" "$(hostname -s)" "$*"; }
[[ -f $ROOT/activate_sana_wam.sh ]] || { log "FATAL: $ROOT/activate_sana_wam.sh missing"; exit 1; }
# shellcheck disable=SC1091
source "$ROOT/activate_sana_wam.sh"
[[ -s ${HF_TOKEN_PATH:-/nonexistent} ]] || { log "FATAL: HF token file ${HF_TOKEN_PATH:-unset} missing"; exit 1; }
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
log "repo=$REPO -> $DEST"
"$SANA_PY" - "$REPO" "$DEST" "${ALLOW:-}" <<'PY'
import hashlib, os, sys, time
from huggingface_hub import HfApi, snapshot_download
repo, dest, allow = sys.argv[1], sys.argv[2], sys.argv[3].split() or None
api = HfApi(); print("token user:", api.whoami().get("name"), flush=True)
info = api.model_info(repo, files_metadata=True)
sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
print(f"repo has {len(sizes)} files, {sum(sizes.values())/1e9:.2f} GB total", flush=True)
for f, n in sorted(sizes.items(), key=lambda kv: -kv[1])[:8]:
    print(f"    {n/1e9:8.2f} GB  {f}", flush=True)
t0 = time.time()
snapshot_download(repo, local_dir=dest, allow_patterns=allow, max_workers=8)
print(f"snapshot_download finished in {time.time()-t0:.0f}s", flush=True)
# Digest sources, in order of preference: a SHA256SUMS file, else a MANIFEST.json carrying a
# "files": [{"name", "sha256"}] list (the shape the Sana-RWM / OpenWAM bundles publish), else sizes.
pairs = []
sums = os.path.join(dest, "SHA256SUMS")
manifest = os.path.join(dest, "MANIFEST.json")
if os.path.exists(sums):
    for line in open(sums):
        line = line.strip()
        if line:
            want, name = line.split(None, 1)
            pairs.append((want, name.strip()))
    print(f"verifying {len(pairs)} files against SHA256SUMS")
elif os.path.exists(manifest):
    import json
    man = json.load(open(manifest))
    pairs = [(f["sha256"], f["name"]) for f in man.get("files", []) if f.get("sha256")]
    print(f"no SHA256SUMS; verifying {len(pairs)} files against MANIFEST.json"
          + (f" (checkpoint {man['checkpoint']})" if "checkpoint" in man else ""))
if not pairs:
    print("no SHA256SUMS and no MANIFEST.json digests -- size check only")
    bad = [f for f, n in sizes.items() if n and os.path.exists(os.path.join(dest, f)) and os.path.getsize(os.path.join(dest, f)) != n]
    print("size mismatches:", bad or "none")
    sys.exit(2 if bad else 0)
bad = []
for want, name in pairs:
    p = os.path.join(dest, name)
    if not os.path.exists(p): bad.append(f"MISSING {name}"); continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""): h.update(chunk)
    got = h.hexdigest()
    status = "OK " if got == want else "BAD"
    if got != want: bad.append(f"SHA MISMATCH {name}: want {want[:16]} got {got[:16]}")
    print(f"  {status} {os.path.getsize(p)/1e9:7.2f} GB  {name}", flush=True)
if bad:
    print("VERIFICATION FAILED:"); [print("   ", b) for b in bad]; sys.exit(3)
print("all files verified against SHA256SUMS")
PY
rc=$?
[[ $rc -eq 0 ]] || { log "FATAL: download/verification failed (rc=$rc)"; exit $rc; }
log "sizes: $(du -sh "$DEST" | cut -f1)"
log "HF REPO READY: $DEST"
