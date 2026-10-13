# shellcheck shell=bash
# Loads the site configuration for every cluster_eval shell entry point: `source "$CE_DIR/lib/site.sh"`.
# Order: an already exported CE_SITE file, else <cluster_eval>/site.env (a copy of, or a symlink to, sites/<site>.env).
# Every CE_* variable is exported afterwards, so python helpers (lib/ce_site.py) and child scripts see the same values.
CE_DIR=${CE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CE_SITE=${CE_SITE:-$CE_DIR/site.env}
if [[ ! -f $CE_SITE ]]; then
  echo "cluster_eval: no site configuration at $CE_SITE -- copy site.env.example to site.env (or ln -s sites/<site>.env site.env," \
       "or export CE_SITE=<file>) and fill it in" >&2
  return 2 2>/dev/null || exit 2
fi
# shellcheck disable=SC1090
source "$CE_SITE"
for _v in CE_SSH_USER CE_SSH_DOMAIN CE_NODES CE_GATEWAY CE_SHARED_HOME CE_REMOTE_INSTALL CE_EVAL_ROOT CE_REMOTE_XPL CE_CKPT_ROOT \
          CE_NODE_ROOT CE_NODE_XPL_WS CE_PY CE_MASTER_ROOT CE_RESULTS_ROOT CE_ROBODOJO_CHECKOUT; do
  [[ -n ${!_v:-} ]] || { echo "cluster_eval: $_v is not set in $CE_SITE" >&2; return 2 2>/dev/null || exit 2; }
done
unset _v
CE_RESERVE=${CE_RESERVE:-}; CE_TAIL_EXCLUDE_NODES=${CE_TAIL_EXCLUDE_NODES:-}; CE_REMOTE_PYDEPS=${CE_REMOTE_PYDEPS:-}
# the XPolicyLab checkout this cluster_eval directory belongs to (scripts/RoboDojo/cluster_eval -> the repo root)
CE_LOCAL_XPL=${CE_LOCAL_XPL:-$(cd "$CE_DIR/../../.." && pwd)}
CE_SSH_OPTS=${CE_SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=15}
export CE_DIR CE_SITE CE_LOCAL_XPL CE_SSH_OPTS $(compgen -v | grep '^CE_')
ce_host(){ echo "$CE_SSH_USER@$1$CE_SSH_DOMAIN"; }        # node name -> ssh target
ce_gw(){ ce_host "$CE_GATEWAY"; }                          # the gateway into the shared home
