#!/usr/bin/env bash
# =============================================================================
# download_partial_ckpt.sh -- assemble an evaluation checkpoint dir from ONE sub-directory of a training-run HF repo
# (work_dir dumps hold many checkpoints: work_dir/checkpoints/<epoch_step>/model/pytorch_model_fsdp.bin).
# Runs on ONE MIT node (shared home). Env:
#   CKPT_REPO   HF repo id, e.g. logits/sana_rwm_policy_from_pretrained_robodojo_320px_1e-4_cosine_clip1_2node
#   CKPT_SUBDIR path inside the repo holding model/pytorch_model_fsdp.bin, e.g. work_dir/checkpoints/epoch_8_step_50000;
#               'auto' (or empty) = the repo root when it holds model/pytorch_model_fsdp.bin, else checkpoints/LATEST of a
#               live resume bundle, else the single */model/pytorch_model_fsdp.bin directory in the repo
#   CKPT_LABEL  name of the assembled snapshot: ~/checkpoints/huggingface/logits/<CKPT_LABEL>
# Result layout (what run_unit*.sh / master.py expect): config.yaml, README.md, model/pytorch_model_fsdp.bin,
# normalization/*.json (+ json_cache/ input cache). config/README/normalization are taken from the repo when present
# (nearest to CKPT_SUBDIR first), otherwise normalization + json_cache are copied from a sibling snapshot whose
# config.yaml uses the SAME training data_dir (recorded in README.md). Re-runnable.
# =============================================================================
set -Eeuo pipefail
ROOT=${ROBODOJO_ROOT:-/dataset2/jiamingt/robodojo}
: "${CKPT_REPO:?CKPT_REPO required}"; CKPT_SUBDIR=${CKPT_SUBDIR:-auto}; : "${CKPT_LABEL:?CKPT_LABEL required}"
CKPT_ROOT=${CKPT_ROOT:-$HOME/checkpoints/huggingface/logits/$CKPT_LABEL}
MIRROR=$HOME/robodojo_install/logs; mkdir -p "$MIRROR" "$CKPT_ROOT"
exec > >(tee -a "$MIRROR/download_partial_ckpt.log") 2>&1
log() { printf '[%s] [%s] %s\n' "$(date '+%F %T')" "$(hostname -s)" "$*"; }
[[ -f $ROOT/activate_sana_wam.sh ]] || { log "FATAL: $ROOT/activate_sana_wam.sh missing"; exit 1; }
# shellcheck disable=SC1091
source "$ROOT/activate_sana_wam.sh"
[[ -s ${HF_TOKEN_PATH:-/nonexistent} ]] || { log "FATAL: HF token file ${HF_TOKEN_PATH:-unset} missing"; exit 1; }
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
log "repo=$CKPT_REPO subdir=$CKPT_SUBDIR -> $CKPT_ROOT"
"$SANA_PY" - "$CKPT_REPO" "$CKPT_SUBDIR" "$CKPT_ROOT" <<'PY'
import glob, hashlib, json, os, shutil, sys, time, datetime
from huggingface_hub import HfApi, hf_hub_download
repo, subdir, root = sys.argv[1], sys.argv[2].strip("/"), sys.argv[3]
api = HfApi(); print("token user:", api.whoami().get("name"), flush=True)
info = api.model_info(repo, files_metadata=True)
files = {s.rfilename: (s.size or 0) for s in info.siblings}
print(f"repo has {len(files)} files; top-level entries: {sorted({f.split('/')[0] for f in files})}", flush=True)
insub = sorted(f for f in files if f.startswith(subdir + "/"))
print(f"files under {subdir}/: {len(insub)}"); [print("   ", f, f"{files[f]/1e9:.2f} GB") for f in insub]
ckpts = sorted({f.rsplit('/', 2)[0] for f in files if f.endswith('/model/pytorch_model_fsdp.bin')})
print("checkpoint dirs in repo:", ckpts, "| root weights:", "model/pytorch_model_fsdp.bin" in files, flush=True)
if subdir in ("", "auto"):
    if "model/pytorch_model_fsdp.bin" in files:
        subdir = ""
    elif "checkpoints/LATEST" in files:
        latest = open(hf_hub_download(repo, "checkpoints/LATEST", local_dir=os.path.join(root, ".hf_partial"))).read().strip()
        subdir = f"checkpoints/{latest}"
    elif len(ckpts) == 1:
        subdir = ckpts[0]
    else:
        sys.exit(f"FATAL: CKPT_SUBDIR=auto cannot pick a checkpoint dir among {ckpts}")
    print("auto-selected checkpoint dir:", repr(subdir), flush=True)
weights = (f"{subdir}/" if subdir else "") + "model/pytorch_model_fsdp.bin"
if weights not in files: sys.exit(f"FATAL: {weights} not in the repo")
def nearest(name):
    cands = [f for f in files if f.endswith("/" + name) or f == name]
    def dist(f):
        d = os.path.dirname(f); p = subdir
        while p and not (d == p or d == "" or p.startswith(d + "/") or d.startswith(p + "/")): p = os.path.dirname(p)
        return (0 if d == subdir else 1 if subdir.startswith(d + "/") or d == "" else 2, len(d))
    return sorted(cands, key=dist)[0] if cands else None
def fetch(remote, local_rel):
    dst = os.path.join(root, local_rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == files.get(remote, -1):
        print("  already present:", local_rel, flush=True); return dst
    t0 = time.time(); p = hf_hub_download(repo, remote, local_dir=os.path.join(root, ".hf_partial"))
    shutil.move(p, dst); print(f"  fetched {remote} -> {local_rel} ({files[remote]/1e9:.2f} GB, {time.time()-t0:.0f}s)", flush=True); return dst
fetch(weights, "model/pytorch_model_fsdp.bin")
cfg_remote = nearest("config.yaml"); print("config.yaml source:", cfg_remote, flush=True)
config_src = os.environ.get("CONFIG_SRC", "")          # '<hf repo>:<path in that repo>' when the checkpoint repo ships no config.yaml
if cfg_remote:
    fetch(cfg_remote, "config.yaml")
elif config_src:
    src_repo, _, src_path = config_src.partition(":")
    t0 = time.time(); got = hf_hub_download(src_repo, src_path, local_dir=os.path.join(root, ".hf_partial"))
    shutil.move(got, os.path.join(root, "config.yaml")); cfg_remote = config_src
    print(f"  fetched config.yaml from CONFIG_SRC {config_src} ({time.time()-t0:.0f}s)", flush=True)
elif os.path.isfile(os.path.join(root, "config.yaml")):
    cfg_remote = "pre-placed <root>/config.yaml"; print("config.yaml: using the pre-placed copy in the snapshot dir", flush=True)
else:
    sys.exit("FATAL: no config.yaml in the repo; pre-place one at <root>/config.yaml or set CONFIG_SRC='<repo>:<path>'")
norm_remote = sorted(f for f in files if (f.startswith("normalization/") or "/normalization/" in f) and f.endswith(".json"))   # root-level or nested
if norm_remote:
    for f in norm_remote: fetch(f, "normalization/" + os.path.basename(f))
readme_remote = nearest("README.md")
provenance = [f"# {os.path.basename(root)}", "", f"Assembled {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC by download_partial_ckpt.sh from HF repo `{repo}`,",
              f"sub-directory `{subdir}` (weights `{weights}`, config `{cfg_remote}`).", ""]
if readme_remote:
    fetch(readme_remote, "README.upstream.md"); provenance.append(f"Upstream README: `{readme_remote}` (copied to README.upstream.md).")
# normalization / json_cache from a sibling snapshot with the same data_dir when the repo ships none
import yaml
cfg = yaml.safe_load(open(os.path.join(root, "config.yaml")))
data_dir = (cfg.get("data") or {}).get("data_dir"); print("config data_dir:", data_dir, flush=True)
provenance.append(f"Training data_dir: `{data_dir}`."); provenance.append(f"Training config: model `{(cfg.get('model') or {}).get('model')}` image_size `{(cfg.get('model') or {}).get('image_size')}`.")
snapshots = os.path.dirname(root.rstrip("/"))
if not glob.glob(os.path.join(root, "normalization", "*.json")):
    donor = None
    for sib in sorted(glob.glob(os.path.join(snapshots, "*"))):
        if os.path.realpath(sib) == os.path.realpath(root) or not os.path.isfile(os.path.join(sib, "config.yaml")): continue
        try: sc = yaml.safe_load(open(os.path.join(sib, "config.yaml")))
        except Exception: continue
        if (sc.get("data") or {}).get("data_dir") == data_dir and glob.glob(os.path.join(sib, "normalization", "*.json")):
            donor = sib; break
    if donor is None: sys.exit("FATAL: repo ships no normalization/*.json and no sibling snapshot with the same data_dir has one")
    os.makedirs(os.path.join(root, "normalization"), exist_ok=True)
    for n in glob.glob(os.path.join(donor, "normalization", "*.json")): shutil.copy2(n, os.path.join(root, "normalization", os.path.basename(n)))
    if os.path.isdir(os.path.join(donor, "json_cache")):
        shutil.copytree(os.path.join(donor, "json_cache"), os.path.join(root, "json_cache"), dirs_exist_ok=True)
    print("normalization (+json_cache) copied from sibling snapshot with the same data_dir:", donor, flush=True)
    provenance.append(f"normalization/ and json_cache/ copied from sibling snapshot `{os.path.basename(donor)}` (same data_dir; the repo ships none).")
open(os.path.join(root, "README.md"), "w").write("\n".join(provenance) + "\n")
shutil.rmtree(os.path.join(root, ".hf_partial"), ignore_errors=True)
# verification (same gates as download_weights.sh)
def sha256_hex(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
ok = True
for rel in ("config.yaml", "README.md", "model/pytorch_model_fsdp.bin"):
    e = os.path.exists(os.path.join(root, rel)); ok &= e; print(("OK  " if e else "MISSING ") + rel, flush=True)
norms = glob.glob(os.path.join(root, "normalization", "*.json")); ok &= bool(norms); print(("OK  " if norms else "MISSING ") + "normalization: " + ", ".join(map(os.path.basename, norms)))
for norm in norms:
    for entry in (json.load(open(norm)).get("input_caches") or []):
        name = os.path.basename(str(entry.get("path", ""))); want = str(entry.get("sha256", "")).split(":")[-1].lower()
        c = os.path.join(root, "json_cache", name)
        good = os.path.isfile(c) and sha256_hex(c) == want
        if not good:   # a byte-identical copy in a sibling snapshot (same corpus, same cache build) is as good as the repo's own
            for cand in sorted(glob.glob(os.path.join(snapshots, "*", "json_cache", name))):
                if os.path.realpath(os.path.dirname(os.path.dirname(cand))) != os.path.realpath(root) and sha256_hex(cand) == want:
                    os.makedirs(os.path.dirname(c), exist_ok=True); shutil.copy2(cand, c); good = True
                    print("OK  normalization input cache copied from sibling snapshot:", cand, flush=True); break
        # only the SANA RPC lane (run_unit.sh) needs json_cache/; the XPolicyLab lane never reads it -> a warning, not a failure
        print(("OK  " if good else "WARNING missing (XPolicyLab lane does not need it) ") + f"normalization input cache {name} sha256 {want[:12]}...", flush=True)
print("sizes: weights %.2f GB" % (os.path.getsize(os.path.join(root, 'model/pytorch_model_fsdp.bin')) / 1e9), flush=True)
if not ok: sys.exit("FATAL: checkpoint dir incomplete")
open(os.path.join(root, ".DONE"), "w").write(datetime.datetime.utcnow().isoformat() + "\n")
print("PARTIAL CHECKPOINT READY:", root, flush=True)
PY
log "done"
