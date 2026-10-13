#!/usr/bin/env bash
# Run on the control host: LABEL=<snapshot> WEIGHTS_SHA=<hex> [CONFIG_SHA=<hex>] [NORM_SHA=<hex>] [HOST=<node>, default CE_GATEWAY] bash verify_ckpt_sha.sh
# sha256-verifies an assembled checkpoint snapshot on the shared NFS home against the digests the upstream repo
# published (download_partial_ckpt.sh checks presence and the normalization input cache, not the weight file itself).
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
: "${LABEL:?LABEL=<snapshot dir name> required}"; : "${WEIGHTS_SHA:?WEIGHTS_SHA=<hex> required}"
HOST=${HOST:-$CE_GATEWAY}; CONFIG_SHA=${CONFIG_SHA:-}; NORM_SHA=${NORM_SHA:-}
ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$(ce_host "$HOST")" "
  set -u
  CK=$CE_CKPT_ROOT/$LABEL
  echo \"== \$CK ==\"; ls -la \"\$CK\" \"\$CK/model\" \"\$CK/normalization\" 2>&1 | sed 's/^/   /'
  check() {  # <file> <expected sha or empty>
    [ -f \"\$1\" ] || { echo \"MISSING \$1\"; return 1; }
    got=\$(sha256sum \"\$1\" | cut -d' ' -f1)
    if [ -z \"\$2\" ]; then echo \"sha256 \$got  \$(basename \"\$1\") (no expectation given)\"; return 0; fi
    if [ \"\$got\" = \"\$2\" ]; then echo \"OK   \$(basename \"\$1\") sha256 \$got\"; else echo \"BAD  \$(basename \"\$1\") want \$2 got \$got\"; return 1; fi
  }
  rc=0
  check \"\$CK/model/pytorch_model_fsdp.bin\" '$WEIGHTS_SHA' || rc=1
  check \"\$CK/config.yaml\" '$CONFIG_SHA' || rc=1
  for n in \"\$CK\"/normalization/*.json; do check \"\$n\" '$NORM_SHA' || rc=1; done
  echo \"== sizes ==\"; du -sh \"\$CK\" \"\$CK/model/pytorch_model_fsdp.bin\"
  [ \$rc -eq 0 ] && echo 'CHECKPOINT SHA VERIFIED' || echo 'CHECKPOINT SHA VERIFICATION FAILED'
  exit \$rc
" 2>&1
