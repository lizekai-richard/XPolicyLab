#!/usr/bin/env bash
# deploy.sh [--dry-run] [--force] -- push this checkout to the cluster's shared home through the gateway (control host). The
# GPU nodes run what lands there; eval/run.sh check-sync refuses a launch while the two differ.
#   eval/                               -> <CE_REMOTE_INSTALL>/eval/     the workers (run_unit_xpl.sh, run_unit.sh, pre_stretch_labels.txt, ...)
#   node/                               -> <CE_REMOTE_INSTALL>/          node setup / checkpoint download / server scripts (top level:
#                                                                        the master calls <CE_REMOTE_INSTALL>/stop_server.sh)
#   the active site file (CE_SITE)      -> <CE_REMOTE_INSTALL>/site.env  what every worker and node script sources
#   this XPolicyLab checkout            -> <CE_REMOTE_XPL>/             tracked files as they are in the working tree (the adapters
#                                                                        the policy servers import)
#   <CE_ROBODOJO_CHECKOUT>/env_cfg/     -> <dirname CE_REMOTE_XPL>/env_cfg/   the parent-workspace env_cfg the adapters resolve
# Then reports what the nodes still need and deploy.sh does not install (eval/RUNBOOK.md, section 1): the node runtime
# ($CE_NODE_ROOT/activate.sh, activate_sana_wam.sh), the pydeps dir, the text encoder and the VAE.
# Files are replaced by rename (rsync's default), so a running worker keeps reading the copy it started with; still, a live
# master of this site would serve its remaining units with the new code, so deploy.sh refuses while one is alive (--force).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/site.sh
source "$HERE/lib/site.sh"
DRY=(); FORCE=0
for a in "$@"; do case $a in --dry-run|-n) DRY=(-n) ;; --force) FORCE=1 ;; *) sed -n 2,16p "$0"; exit 2 ;; esac; done
GW=$(ce_gw); RSH="ssh $CE_SSH_OPTS"
EXCL=(--exclude __pycache__ --exclude '*.pyc' --exclude '*.bak*' --exclude .pytest_cache --exclude logs/)

live=$("$CE_PY" - "$CE_MASTER_ROOT" <<'PYEOF'
import os, pathlib, sys
for pidf in sorted(pathlib.Path(sys.argv[1]).glob("*/master.pid")):
    try: os.kill(int(pidf.read_text().strip()), 0)
    except Exception: continue
    print(pidf.parent.name)
PYEOF
)
if [[ -n $live && $FORCE != 1 && ${#DRY[@]} -eq 0 ]]; then
  echo "REFUSE: live masters of this site would serve their remaining units with the new code (--force to deploy anyway):"
  sed 's/^/  /' <<<"$live"; exit 3
fi
echo "deploying $CE_DIR ($(git -C "$CE_LOCAL_XPL" rev-parse --short HEAD)$(git -C "$CE_LOCAL_XPL" diff --quiet HEAD -- . || echo '+uncommitted')) to $GW (site $CE_SITE)${DRY:+ [dry run]}"
[[ ${#DRY[@]} -gt 0 ]] || ssh -n $CE_SSH_OPTS "$GW" "mkdir -p '$CE_REMOTE_INSTALL/eval' '$CE_REMOTE_INSTALL/logs' '$CE_REMOTE_XPL' '$(dirname "$CE_REMOTE_XPL")/env_cfg' '$CE_EVAL_ROOT' '$CE_CKPT_ROOT'"
sync_step(){ # LABEL RSYNC-ARGS... : one rsync, its changed items summarized
  local label=$1 out; shift
  out=$(rsync -a -i "${DRY[@]}" -e "$RSH" "$@")
  printf '  %-12s %4d changed item(s)%s\n' "$label" "$(grep -c . <<<"$out" || true)" "$(grep -m3 '^[<>c*]' <<<"$out" | awk '{printf " %s", $2}')"
}
sync_step eval/ "${EXCL[@]}" "$CE_DIR/eval/" "$GW:$CE_REMOTE_INSTALL/eval/"
sync_step node/ "${EXCL[@]}" --exclude README.md "$CE_DIR/node/" "$GW:$CE_REMOTE_INSTALL/"
sync_step site.env -L "$CE_SITE" "$GW:$CE_REMOTE_INSTALL/site.env"
sync_step XPolicyLab --from0 --files-from=<(git -C "$CE_LOCAL_XPL" ls-files -z) "$CE_LOCAL_XPL/" "$GW:$CE_REMOTE_XPL/"
sync_step env_cfg/ "$CE_ROBODOJO_CHECKOUT/env_cfg/" "$GW:$(dirname "$CE_REMOTE_XPL")/env_cfg/"

echo "== what the nodes need that deploy.sh does not install (checked through $CE_GATEWAY; RUNBOOK section 1)"
ssh -n $CE_SSH_OPTS "$GW" "
  ok(){ printf '  %-4s %s\n' \"\$1\" \"\$2\"; }
  [ -f '$CE_NODE_ROOT/activate.sh' ] && ok OK '$CE_NODE_ROOT/activate.sh (on $CE_GATEWAY; every node needs its own)' || ok MISS '$CE_NODE_ROOT/activate.sh: node/install_node.sh'
  [ -f '$CE_NODE_ROOT/activate_sana_wam.sh' ] && ok OK '$CE_NODE_ROOT/activate_sana_wam.sh' || ok MISS '$CE_NODE_ROOT/activate_sana_wam.sh: node/install_sana_wam.sh'
  if [ -n '$CE_REMOTE_PYDEPS' ]; then [ -d '$CE_REMOTE_PYDEPS/websockets' ] && [ -d '$CE_REMOTE_PYDEPS/msgpack_numpy' -o -f '$CE_REMOTE_PYDEPS/msgpack_numpy.py' ] && ok OK '$CE_REMOTE_PYDEPS' || ok MISS '$CE_REMOTE_PYDEPS: websockets msgpack msgpack-numpy h5py for the policy-server python (RUNBOOK 1.4)'; fi
  [ -f '${CE_TEXT_ENCODER_PATH:-/nonexistent}/config.json' ] && ok OK '$CE_TEXT_ENCODER_PATH' || ok MISS 'text encoder ${CE_TEXT_ENCODER_PATH:-<CE_TEXT_ENCODER_PATH unset>} (policy/SANA_WAM/INSTALLATION.md 2b)'
  [ -f '${CE_VAE_PATH:-/nonexistent}/vae/config.json' ] && ok OK '$CE_VAE_PATH' || ok MISS 'VAE ${CE_VAE_PATH:-<CE_VAE_PATH unset>}/vae (policy/SANA_WAM/INSTALLATION.md 2a)'
  true"
echo "done; verify with: bash $CE_DIR/eval/run.sh check-sync   (CKPT_LABEL may stay unset)"
