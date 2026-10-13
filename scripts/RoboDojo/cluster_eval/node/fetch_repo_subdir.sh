#!/usr/bin/env bash
# fetch_repo_subdir.sh -- download every file under ONE sub-directory of an HF repo into a directory on the shared home,
# WITHOUT touching anything else (staging area for artifacts uploaded after a snapshot was assembled, e.g. normalization/).
# Runs on ONE node. Env: CKPT_REPO (HF repo id), REPO_SUBDIR (e.g. normalization), DEST (absolute dir on the shared home).
# Prints size + sha256 per file; terminal line "FETCH DONE <n files>" or "FATAL ...". Re-runnable (overwrites DEST files).
set -Eeuo pipefail
_CE_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
[[ -f $_CE_HERE/site.env ]] && source "$_CE_HERE/site.env"   # the site configuration deploy.sh copies next to the node scripts
ROOT=${ROBODOJO_ROOT:-${CE_NODE_ROOT:?no site.env next to this script (deploy.sh) and no ROBODOJO_ROOT}}
: "${CKPT_REPO:?CKPT_REPO required}"; : "${REPO_SUBDIR:?REPO_SUBDIR required}"; : "${DEST:?DEST required}"
mkdir -p "$DEST" "$_CE_HERE/logs"
exec > >(tee -a "$_CE_HERE/logs/fetch_repo_subdir.log") 2>&1
log() { printf '[%s] [%s] %s\n' "$(date '+%F %T')" "$(hostname -s)" "$*"; }
[[ -f $ROOT/activate_sana_wam.sh ]] || { log "FATAL: $ROOT/activate_sana_wam.sh missing"; exit 1; }
# shellcheck disable=SC1091
source "$ROOT/activate_sana_wam.sh"
[[ -s ${HF_TOKEN_PATH:-/nonexistent} ]] || { log "FATAL: HF token file ${HF_TOKEN_PATH:-unset} missing"; exit 1; }
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
log "repo=$CKPT_REPO subdir=$REPO_SUBDIR -> $DEST"
"$SANA_PY" - "$CKPT_REPO" "$REPO_SUBDIR" "$DEST" <<'PY'
import hashlib, os, shutil, sys, time
from huggingface_hub import HfApi, hf_hub_download
repo, subdir, dest = sys.argv[1], sys.argv[2].strip("/"), sys.argv[3]
api = HfApi(); print("token user:", api.whoami().get("name"), flush=True)
info = api.model_info(repo, files_metadata=True)
files = {s.rfilename: (s.size or 0) for s in info.siblings}
hits = sorted(f for f in files if f == subdir or f.startswith(subdir + "/") or f"/{subdir}/" in f)
print(f"repo has {len(files)} files; {len(hits)} under or named '{subdir}': {hits}", flush=True)
if not hits: sys.exit(f"FATAL: nothing under {subdir}/ in {repo}")
stage = os.path.join(dest, ".hf_stage"); n = 0
for f in hits:
    t0 = time.time(); p = hf_hub_download(repo, f, local_dir=stage)
    out = os.path.join(dest, f); os.makedirs(os.path.dirname(out), exist_ok=True); shutil.copy2(p, out); n += 1
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"  {f}  {os.path.getsize(out)} bytes  sha256 {sha}  ({time.time()-t0:.0f}s) -> {out}", flush=True)
shutil.rmtree(stage, ignore_errors=True)
print(f"FETCH DONE {n} files", flush=True)
PY
log "done"
