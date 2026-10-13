#!/bin/bash
# run_learned_server_only_local.sh -- run_learned_server_only.sbatch without Slurm.
#
# Same gates, manifest lanes, server invocation, endpoint gate, handoff bundle
# and evidence layout as dev/rwm/simulation_evaluation/robodojo/launch/
# run_learned_server_only.sbatch (Sana rwm/zekai @ bd9bf130d).  Only the Slurm
# plumbing differs:
#   * no #SBATCH / srun / scancel: the server is a plain child process on the
#     GPU named by SANA_SERVER_GPU (default 0); stop the session with
#     `kill -TERM $(cat <evidence>/launcher.pid)`
#   * the evidence dir may be any fresh absolute path containing "robodojo";
#     default $HOME/robodojo_evidence/robodojo_server_<ts>_<pid>, i.e. on the
#     shared home so every lN node sees the client handoff bundle
#   * defaults point at the MIT layout (~/Sana, /dataset2/jiamingt/robodojo) and
#     the DEFAULT CHECKPOINT IS A DIFFERENT MODEL than the sbatch's: the 320px
#     epoch4/step25000 snapshot fetched by download_weights.sh (its normalization
#     file carries the same name robodojo_arx_x5_model_fps_25_f25_normalization.json);
#     every default stays env-overridable (SANA_POLICY_CHECKPOINT_ROOT, ...)
#   * SANA_EVIDENCE_ROOT (default $HOME/robodojo_evidence) hosts the default dir
#   * the script cd's into SANA_REPO (the sbatch relied on being submitted from
#     the repo root: gemma-2-2b-it-local is a repo-relative path)
#   * SANA_DRY_RUN=1 stops after preflight + manifest, before the server starts
# usage: bash run_learned_server_only_local.sh [ABSOLUTE_FRESH_EVIDENCE_DIR]
# Launch it detached (start_server.sh does `nohup setsid ... &`): run from a
# plain SSH session, an SSH drop would HUP both this script and the server.

# Cluster half of the split RoboDojo evaluation: this job runs ONLY the SANA
# policy server (no simulator client, no websocket responder) and publishes
# a client handoff bundle under the evidence directory.  The simulator
# client runs on a separate RT-capable machine and dials the RPC endpoint
# through an SSH tunnel, so the frozen session config advertises 127.0.0.1:
# the remote client connects to its own loopback tunnel entrance, never to
# the compute node address directly.
#
# The peer of this server is the REAL simulator client, so the manifest lane
# freezes the official task instruction exactly as the simulator smoke does
# (byte-equal to the live DescManager rendering, gated below), not the debug
# harness literal.
#
# After the endpoint gate passes, this job IS the server: it waits on the
# server process until the server exits on its own, the operator cancels the
# job, or the partition wall clock ends it.  There is no client verdict on
# this side; run_status.json records only the server lifecycle.
#
# Three manifest lanes share this launcher:
#   * single-task (SANA_TASK_SET unset): the pinned-task lane, unchanged.
#   * explicit task list (SANA_TASK_LIST_FILE=<file> with
#     SANA_EPISODES_PER_TASK=N): one task name per line (blank lines and
#     #comments ignored), each validated against the derived universe, the
#     single-template gate, AND the dual_x5 embodiment (the served
#     checkpoint family cannot drive the franka-competition tasks).  The
#     lane label is list-<sha8 of the file bytes>.
#   * task-set benchmark (SANA_TASK_SET=A|B with SANA_EPISODES_PER_TASK=N):
#     the runnable task universe is mechanically re-derived from the pinned
#     checkout's task sources, tasks without a single placeholder-free
#     gen_instruction template are excluded (disclosed below), the remaining
#     bytewise-ascending task list is split into two contiguous halves (A =
#     first half, B = second half), and each selected task contributes its
#     FIRST N frozen layouts.  The handoff additionally ships
#     benchmark_plan.json, the ordered replay plan the client runner mirrors
#     verbatim.  Two server jobs, one per task set, cover the benchmark.

set -euo pipefail
umask 077
LAUNCHER_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)

if [[ "$#" -gt 1 ]]; then
  echo "usage: bash run_learned_server_only_local.sh [ABSOLUTE_FRESH_EVIDENCE_DIR]" >&2
  exit 64
fi
if [[ "$#" -eq 1 ]]; then
  EVIDENCE_DIR=$(realpath -m -- "$1")
else
  EVIDENCE_DIR=$(realpath -m -- "${SANA_EVIDENCE_ROOT:-${HOME}/robodojo_evidence}/robodojo_server_$(date +%Y%m%d-%H%M%S)_$$")
fi
if [[ "${EVIDENCE_DIR}" != /* || "${EVIDENCE_DIR}" != *robodojo* ]]; then
  echo "evidence directory must be an absolute path containing 'robodojo'" >&2
  exit 64
fi
if [[ -e "${EVIDENCE_DIR}" ]]; then
  echo "evidence directory must not already exist: ${EVIDENCE_DIR}" >&2
  exit 73
fi
mkdir -p -- "${EVIDENCE_DIR}"
printf '%s\n' "$$" >"${EVIDENCE_DIR}/launcher.pid"

# The server budget covers the eager checkpoint load; there is no client
# watchdog in this launcher because no client runs inside this job.
SERVER_STARTUP_TIMEOUT_SECONDS=${ROBODOJO_SERVER_STARTUP_TIMEOUT_SECONDS:-1500}
SERVER_SHUTDOWN_GRACE_SECONDS=${ROBODOJO_SERVER_SHUTDOWN_GRACE_SECONDS:-90}
PREFLIGHT_WATCHDOG_SECONDS=${ROBODOJO_PREFLIGHT_WATCHDOG_SECONDS:-600}
for timeout_name in \
  SERVER_STARTUP_TIMEOUT_SECONDS SERVER_SHUTDOWN_GRACE_SECONDS \
  PREFLIGHT_WATCHDOG_SECONDS; do
  if [[ ! "${!timeout_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${timeout_name} must be a positive integer" >&2
    exit 64
  fi
done

SANA_REPO=${SANA_REPO:-${HOME}/Sana}
SANA_PY=${SANA_PY:-${ROBODOJO_ROOT:-/dataset2/jiamingt/robodojo}/envs/sana-wam/bin/python}
ROBODOJO_CHECKOUT=${ROBODOJO_CHECKOUT:-${ROBODOJO_ROOT:-/dataset2/jiamingt/robodojo}/RoboDojo-official-clone-20260823}
ROBODOJO_EXPECTED_COMMIT=${ROBODOJO_EXPECTED_COMMIT:-2184bf8844ea9d205382c4aefa3a694311418251}
ROBODOJO_XPOLICYLAB_EXPECTED_COMMIT=${ROBODOJO_XPOLICYLAB_EXPECTED_COMMIT:-432f82b1758c5b1202e42a3dfe014546dbc50871}
CKPT_ROOT=${SANA_POLICY_CHECKPOINT_ROOT:-${HOME}/checkpoints/huggingface/logits/sana_rwm_policy_from_pretrained_robodojo_320px_epoch4_step25000}
CONFIG=${SANA_POLICY_CONFIG:-${CKPT_ROOT}/config.yaml}
TRAINING_LOG=${SANA_POLICY_TRAINING_LOG:-${CKPT_ROOT}/README.md}
NORMALIZATION=${SANA_POLICY_NORMALIZATION:-${CKPT_ROOT}/normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json}
CHECKPOINT_INPUT=${SANA_POLICY_CHECKPOINT:-${CKPT_ROOT}/model/pytorch_model_fsdp.bin}
CHECKPOINT=$(realpath -e -- "${CHECKPOINT_INPUT}")
# The exported training config names the Hugging Face hub id for the VAE, so
# the local weights directory is supplied explicitly.
VAE_PRETRAINED=${SANA_POLICY_VAE_PRETRAINED:-${SANA_REPO}/output/pretrained_models/LTX-2.3-Diffusers}
if [[ ! -e "${SANA_REPO}/.git" ]]; then
  echo "SANA_REPO is not a git checkout: ${SANA_REPO}" >&2
  exit 66
fi
# The GPU this server process owns (nodes carry eight); the simulator client
# should be given a different index.
SANA_SERVER_GPU=${SANA_SERVER_GPU:-0}
if [[ ! "${SANA_SERVER_GPU}" =~ ^[0-9]+$ ]]; then
  echo "SANA_SERVER_GPU must be a non-negative integer GPU index" >&2
  exit 64
fi
if command -v nvidia-smi >/dev/null 2>&1 \
    && ! nvidia-smi -i "${SANA_SERVER_GPU}" --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
  echo "SANA_SERVER_GPU=${SANA_SERVER_GPU} is not a visible GPU index on $(hostname -s)" >&2
  exit 66
fi
RUNTIME_COMMIT=${SANA_RUNTIME_COMMIT:-$(git -C "${SANA_REPO}" rev-parse HEAD)}

# Tunnel-first networking knobs.  The server binds every interface so the
# tunnel target (this compute node) is reachable from the login node, while
# the session config advertises 127.0.0.1 because the remote client dials
# its local SSH tunnel entrance.  The server main derives
# advertise_host = rpc_host whenever rpc_host differs from bind_host, and
# ServerLaunchConfig rejects a wildcard bind without an advertise host, so a
# wildcard bind requires an advertise host different from the bind host.
SANA_RPC_BIND_HOST=${SANA_RPC_BIND_HOST:-0.0.0.0}
SANA_RPC_ADVERTISE_HOST=${SANA_RPC_ADVERTISE_HOST:-127.0.0.1}
ARRAY_OFFSET=${SLURM_ARRAY_TASK_ID:-0}
JOB_TOKEN=${SLURM_JOB_ID:-$$}
SANA_RPC_PORT=${SANA_RPC_PORT:-$((40000 + (JOB_TOKEN + ARRAY_OFFSET) % 20000))}
if [[ ! "${SANA_RPC_PORT}" =~ ^[1-9][0-9]*$ ]] || ((SANA_RPC_PORT > 65535)); then
  echo "SANA_RPC_PORT must be an integer in [1, 65535]" >&2
  exit 64
fi
if [[ -z "${SANA_RPC_BIND_HOST}" || -z "${SANA_RPC_ADVERTISE_HOST}" ]]; then
  echo "SANA_RPC_BIND_HOST and SANA_RPC_ADVERTISE_HOST must be non-empty" >&2
  exit 64
fi
if [[ ( "${SANA_RPC_BIND_HOST}" == 0.0.0.0 || "${SANA_RPC_BIND_HOST}" == :: ) \
    && "${SANA_RPC_ADVERTISE_HOST}" == "${SANA_RPC_BIND_HOST}" ]]; then
  echo "a wildcard bind requires an advertise host different from the bind host" >&2
  exit 64
fi

# Manifest lane selector.  SANA_TASK_SET unset keeps the single-task lane;
# A|B selects one contiguous half of the derived benchmark task list and
# requires an explicit per-task episode budget.
TASK_SET=${SANA_TASK_SET:-}
TASK_LIST_FILE=${SANA_TASK_LIST_FILE:-}
if [[ -n "${TASK_SET}" && -n "${TASK_LIST_FILE}" ]]; then
  echo "SANA_TASK_SET and SANA_TASK_LIST_FILE are mutually exclusive" >&2
  exit 64
fi
if [[ -n "${TASK_LIST_FILE}" ]]; then
  if [[ ! -f "${TASK_LIST_FILE}" ]]; then
    echo "SANA_TASK_LIST_FILE does not name a file: ${TASK_LIST_FILE}" >&2
    exit 66
  fi
  # The label doubles as the plan's task_set name; hashing the file bytes
  # makes reruns of the same list self-identifying.
  TASK_LIST_SHA=$(sha256sum -- "${TASK_LIST_FILE}")
  TASK_SET=list-${TASK_LIST_SHA:0:8}
fi
if [[ -n "${TASK_SET}" ]]; then
  if [[ -z "${TASK_LIST_FILE}" && "${TASK_SET}" != A && "${TASK_SET}" != B ]]; then
    echo "SANA_TASK_SET must be A or B" >&2
    exit 64
  fi
  if [[ ! "${SANA_EPISODES_PER_TASK:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SANA_EPISODES_PER_TASK must be a positive integer when SANA_TASK_SET or SANA_TASK_LIST_FILE is set" >&2
    exit 64
  fi
  EPISODES_PER_TASK=${SANA_EPISODES_PER_TASK}
  if [[ -n "${SANA_EVAL_TASK:-}" || -n "${SANA_EVAL_INSTRUCTION:-}" ]]; then
    echo "the multi-task lanes derive tasks and instructions themselves; unset SANA_EVAL_TASK and SANA_EVAL_INSTRUCTION" >&2
    exit 64
  fi
elif [[ -n "${SANA_EPISODES_PER_TASK:-}" ]]; then
  echo "SANA_EPISODES_PER_TASK requires SANA_TASK_SET or SANA_TASK_LIST_FILE" >&2
  exit 64
fi

# Pinned episode identity for the frozen manifest.  insert_key at eval seed 0
# is the default so the manifest lane stays directly comparable with the
# in-cluster smokes; a non-default task must supply its own official
# instruction literal, which the drift gate below verifies against the
# upstream task source exactly like the simulator smoke.
CONFIG_NAME=arx_x5
EVAL_SEED=${SANA_EVAL_SEED:-0}
if [[ ! "${EVAL_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SANA_EVAL_SEED must be a non-negative integer" >&2
  exit 64
fi
if [[ -z "${TASK_SET}" ]]; then
  TASK=${SANA_EVAL_TASK:-insert_key}
  DEFAULT_INSTRUCTION='Pick up the key, hand it over to the other hand, insert it into the keyhole, then turn it.'
  if [[ "${TASK}" == insert_key ]]; then
    INSTRUCTION=${SANA_EVAL_INSTRUCTION:-${DEFAULT_INSTRUCTION}}
  else
    if [[ -z "${SANA_EVAL_INSTRUCTION:-}" ]]; then
      echo "SANA_EVAL_INSTRUCTION must carry the official instruction literal for task ${TASK}" >&2
      exit 64
    fi
    INSTRUCTION=${SANA_EVAL_INSTRUCTION}
  fi
  STATUS_TASK=${TASK}
else
  STATUS_TASK=task-set-${TASK_SET}
fi
INFERENCE_NUM_FRAMES=${SANA_POLICY_INFERENCE_NUM_FRAMES:-25}
N_ACTION_STEPS=${SANA_POLICY_N_ACTION_STEPS:-24}
EXECUTED_STEPS_PER_CHUNK=${SANA_POLICY_EXECUTED_STEPS_PER_CHUNK:-${N_ACTION_STEPS}}
for horizon_name in INFERENCE_NUM_FRAMES N_ACTION_STEPS EXECUTED_STEPS_PER_CHUNK; do
  if [[ ! "${!horizon_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${horizon_name} must be a positive integer" >&2
    exit 64
  fi
done
if [[ "${INFERENCE_NUM_FRAMES}" -ne 25 \
    || "${N_ACTION_STEPS}" -ne 24 \
    || "${EXECUTED_STEPS_PER_CHUNK}" -ne 24 ]]; then
  echo "this learned-policy server is frozen to F25/K24/E24" >&2
  exit 64
fi
RUN_ID=robodojo-server-only-${JOB_TOKEN}-${ARRAY_OFFSET}
NODE_FQDN=${SANA_NODE_FQDN:-$(hostname -f 2>/dev/null || hostname)}
# Server-side diagnostic prediction videos (one MP4 + receipt per inference).
# Decoded pixels stay out of the RPC; the cost is one VAE decode per replan —
# and that decode + MP4 encode runs INSIDE the timed inference call, so
# SANA_PREDICTION_VIDEOS=0 (no --prediction-video-dir) trades the diagnostic
# videos for a shorter per-chunk latency without touching the policy numerics.
PREDICTION_VIDEO_DIR=${SANA_PREDICTION_VIDEO_DIR:-${EVIDENCE_DIR}/prediction_videos}
SANA_PREDICTION_VIDEOS=${SANA_PREDICTION_VIDEOS:-1}
case "${SANA_PREDICTION_VIDEOS}" in 0|1) ;; *) echo "SANA_PREDICTION_VIDEOS must be 0 or 1" >&2; exit 64;; esac
PREDICTION_VIDEO_ARGS=()
if [[ "${SANA_PREDICTION_VIDEOS}" == 1 ]]; then
  PREDICTION_VIDEO_ARGS=(--prediction-video-dir "${PREDICTION_VIDEO_DIR}")
fi

MANIFEST=${EVIDENCE_DIR}/preflight_manifest.json
INSTRUCTIONS_FILE=${EVIDENCE_DIR}/instructions.json
BENCHMARK_PLAN=${EVIDENCE_DIR}/benchmark_plan.json
TASK_SPLIT_FILE=${EVIDENCE_DIR}/task_split.json
SELECTED_TASKS_FILE=${EVIDENCE_DIR}/selected_tasks.txt
ARTIFACT_DIR=${EVIDENCE_DIR}/server_artifacts
FROZEN_TRAINING_LOG=${EVIDENCE_DIR}/training_log_snapshot.input.log
HANDOFF_DIR=${EVIDENCE_DIR}/client_handoff
# Repo-relative entry points of the remote half of this split; the handoff
# README names one of them for the operator on the client machine depending
# on the manifest lane.
REMOTE_CLIENT_SCRIPT=dev/rwm/simulation_evaluation/robodojo/launch/remote_client/run_remote_sim_client.sh
REMOTE_BENCHMARK_CLIENT_SCRIPT=dev/rwm/simulation_evaluation/robodojo/launch/remote_client/run_remote_benchmark_client.sh

required_inputs=(
  "${SANA_PY}"
  "${CONFIG}" "${CHECKPOINT}" "${VAE_PRETRAINED}"
  "${TRAINING_LOG}" "${NORMALIZATION}"
  "${ROBODOJO_CHECKOUT}/env_cfg/${CONFIG_NAME}.yml"
)
if [[ -z "${TASK_SET}" ]]; then
  required_inputs+=(
    "${ROBODOJO_CHECKOUT}/task/RoboDojo/config/${TASK}.yml"
    "${ROBODOJO_CHECKOUT}/task/RoboDojo/tasks/${TASK}.py"
  )
else
  required_inputs+=("${ROBODOJO_CHECKOUT}/task/RoboDojo/tasks")
fi
for required in "${required_inputs[@]}"; do
  if [[ ! -e "${required}" ]]; then
    echo "required runtime input is missing: ${required}" >&2
    exit 66
  fi
done
if [[ ! -x "${SANA_PY}" ]]; then
  echo "SANA_PY must be executable" >&2
  exit 66
fi
# The preflight manifest builder enumerates the frozen layout files to
# freeze the episode identity; the render-only asset subtrees (Robots,
# Object, Material) belong to the simulator machine and are deliberately NOT
# gated on this server-only side.
if [[ ! -d "${ROBODOJO_CHECKOUT}/Assets/Eval_Layout" ]]; then
  echo "RoboDojo assets subtree is missing: Assets/Eval_Layout" >&2
  exit 66
fi
if [[ -z "${TASK_SET}" ]]; then
  if [[ ! -f "${ROBODOJO_CHECKOUT}/Assets/Eval_Layout/RoboDojo/${CONFIG_NAME}/${EVAL_SEED}/${TASK}_0.json" ]]; then
    echo "frozen layout 0 for ${TASK} at eval seed ${EVAL_SEED} is missing" >&2
    exit 66
  fi
else
  # Per-task layout population gates live in the manifest builder, which
  # fails closed on any selected task with no frozen layouts or fewer than
  # the requested episode budget.
  if [[ ! -d "${ROBODOJO_CHECKOUT}/Assets/Eval_Layout/RoboDojo/${CONFIG_NAME}/${EVAL_SEED}" ]]; then
    echo "frozen eval layout directory for eval seed ${EVAL_SEED} is missing" >&2
    exit 66
  fi
fi
if [[ "$(git -C "${SANA_REPO}" rev-parse HEAD)" != "${RUNTIME_COMMIT}" ]]; then
  echo "SANA HEAD does not match SANA_RUNTIME_COMMIT" >&2
  exit 65
fi
ROBODOJO_COMMIT=$(git -C "${ROBODOJO_CHECKOUT}" rev-parse HEAD)
if [[ "${ROBODOJO_COMMIT}" != "${ROBODOJO_EXPECTED_COMMIT}" ]]; then
  echo "RoboDojo checkout commit drifted: ${ROBODOJO_COMMIT}" >&2
  exit 65
fi
XPOLICYLAB_COMMIT=$(git -C "${ROBODOJO_CHECKOUT}/XPolicyLab" rev-parse HEAD)
if [[ "${XPOLICYLAB_COMMIT}" != "${ROBODOJO_XPOLICYLAB_EXPECTED_COMMIT}" ]]; then
  echo "XPolicyLab submodule commit drifted: ${XPOLICYLAB_COMMIT}" >&2
  exit 65
fi

# The frozen manifest instruction must byte-equal the live instruction the
# remote simulator's DescManager renders from the task's gen_instruction
# template; the session reconciles the live obs instruction against the
# frozen manifest before every episode, so a drift here is a guaranteed
# run-killing identity error on the client machine.  Like the simulator
# smoke, this assumes a single placeholder-free gen_instruction template.
# The per-episode step budget belongs to the remote client launcher, so no
# step_lim pin is gated here.  The task-set lane runs the same byte-parity
# gate per task inside the extraction step below.
if [[ -z "${TASK_SET}" ]] && ! grep -Fq "${INSTRUCTION}" \
    "${ROBODOJO_CHECKOUT}/task/RoboDojo/tasks/${TASK}.py"; then
  echo "pinned instruction literal is absent from tasks/${TASK}.py" >&2
  exit 65
fi

# Freeze the training receipt bytes before the server hashes the checkpoint;
# the server checksum-binds this immutable copy.
cp --reflink=auto --preserve=timestamps -- \
  "${TRAINING_LOG}" "${FROZEN_TRAINING_LOG}"
FROZEN_TRAINING_LOG_HEX=$(sha256sum "${FROZEN_TRAINING_LOG}")
FROZEN_TRAINING_LOG_HEX=${FROZEN_TRAINING_LOG_HEX%% *}

{
  printf 'sana_repo=%s\n' "${SANA_REPO}"
  printf 'sana_runtime_commit=%s\n' "${RUNTIME_COMMIT}"
  printf 'robodojo_checkout=%s\n' "${ROBODOJO_CHECKOUT}"
  printf 'robodojo_commit=%s\n' "${ROBODOJO_COMMIT}"
  printf 'xpolicylab_commit=%s\n' "${XPOLICYLAB_COMMIT}"
  printf 'checkpoint=%s\n' "${CHECKPOINT}"
  printf 'normalization=%s\n' "${NORMALIZATION}"
  printf 'training_log_source=%s\n' "${TRAINING_LOG}"
  printf 'training_log_frozen=%s\n' "${FROZEN_TRAINING_LOG}"
  printf 'training_log_frozen_sha256=sha256:%s\n' "${FROZEN_TRAINING_LOG_HEX}"
  if [[ -z "${TASK_SET}" ]]; then
    printf 'task=%s\n' "${TASK}"
  else
    printf 'task_set=%s\n' "${TASK_SET}"
    printf 'episodes_per_task=%s\n' "${EPISODES_PER_TASK}"
  fi
  printf 'config_name=%s\n' "${CONFIG_NAME}"
  printf 'eval_seed=%s\n' "${EVAL_SEED}"
  if [[ -z "${TASK_SET}" ]]; then
    printf 'instruction=%s\n' "${INSTRUCTION}"
  fi
  printf 'inference_num_frames=%s\n' "${INFERENCE_NUM_FRAMES}"
  printf 'n_action_steps=%s\n' "${N_ACTION_STEPS}"
  printf 'compute_node=%s\n' "${NODE_FQDN}"
  printf 'rpc_bind_host=%s\n' "${SANA_RPC_BIND_HOST}"
  printf 'rpc_advertise_host=%s\n' "${SANA_RPC_ADVERTISE_HOST}"
  printf 'rpc_port=%s\n' "${SANA_RPC_PORT}"
} >"${EVIDENCE_DIR}/resolved_inputs.txt"

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
# Policy acceleration preset for the real backend session (policy_runtime), ported from the
# rwm/zekai-merge sbatch: off = eager (historical eval posture, default); fused = the deployment
# hand-fused kernel stack (~3.7x per chunk, kernel-level numerics bands); fused_fp8 = fused + e4m3
# GEMMs (~4.5x, explicit numerics change). Success-rate lineage is NOT comparable across presets.
# fused_portable = the stack minus the SM90-only cores (for the A6000s, SM 8.6); its own numerics band.
export RWM_EVAL_POLICY_ACCELERATION=${RWM_EVAL_POLICY_ACCELERATION:-off}
case "${RWM_EVAL_POLICY_ACCELERATION}" in off|fused|fused_fp8|fused_portable) ;; *)
  echo "RWM_EVAL_POLICY_ACCELERATION must be off, fused, fused_fp8 or fused_portable (got '${RWM_EVAL_POLICY_ACCELERATION}')" >&2; exit 64;; esac
printf 'policy_acceleration=%s\n' "${RWM_EVAL_POLICY_ACCELERATION}" >>"${EVIDENCE_DIR}/resolved_inputs.txt"
printf 'prediction_videos=%s\n' "${SANA_PREDICTION_VIDEOS}" >>"${EVIDENCE_DIR}/resolved_inputs.txt"

# Freeze the manifest instruction mapping, then the layout population.  The
# builder enumerates every frozen layout of each evaluated task exactly like
# the upstream SeedManager; the remote client consumes manifest ordinals from
# 0 in the upstream seed-queue order.  The frozen instruction literals are
# the same disclosed simplification as the simulator smoke: the builder is
# simulator-free, so instructions are supplied as final text (byte parity
# with the live rendering is guaranteed by the gen_instruction gates).
if [[ -z "${TASK_SET}" ]]; then
  "${SANA_PY}" -c '
import json, sys
output, task, instruction = sys.argv[1:4]
with open(output, "x", encoding="utf-8") as stream:
    json.dump({task: instruction}, stream, allow_nan=False, indent=2, sort_keys=True)
    stream.write("\n")
' "${INSTRUCTIONS_FILE}" "${TASK}" "${INSTRUCTION}"

  timeout --signal=TERM --kill-after=30s "${PREFLIGHT_WATCHDOG_SECONDS}" \
    env PYTHONNOUSERSITE=1 PYTHONPATH="${SANA_REPO}" \
    "${SANA_PY}" -m \
    dev.rwm.simulation_evaluation.robodojo.common.preflight_manifest \
    --assets-root "${ROBODOJO_CHECKOUT}/Assets" \
    --config-name "${CONFIG_NAME}" \
    --eval-seed "${EVAL_SEED}" \
    --task "${TASK}" \
    --instructions "${INSTRUCTIONS_FILE}" \
    --output "${MANIFEST}" \
    --policy-session-id "${RUN_ID}" \
    >"${EVIDENCE_DIR}/manifest_preflight.stdout.log" \
    2>"${EVIDENCE_DIR}/manifest_preflight.stderr.log"
else
  # Task-set lane: mechanically re-derive the benchmark task universe from
  # the pinned checkout, extract each task's single placeholder-free
  # gen_instruction template with the same verbatim byte-parity discipline
  # as the single-task grep gate, split the ordered list into two contiguous
  # halves, and freeze this job's half.  Any task outside the disclosed
  # exclusion set that fails the single-template gate aborts the job here,
  # before the server starts.
  timeout --signal=TERM --kill-after=30s "${PREFLIGHT_WATCHDOG_SECONDS}" \
    "${SANA_PY}" - \
    "${ROBODOJO_CHECKOUT}/task/RoboDojo/tasks" \
    "${TASK_SET}" "${INSTRUCTIONS_FILE}" "${TASK_SPLIT_FILE}" \
    "${SELECTED_TASKS_FILE}" "${TASK_LIST_FILE}" \
    >"${EVIDENCE_DIR}/task_split_preflight.stdout.log" \
    2>"${EVIDENCE_DIR}/task_split_preflight.stderr.log" <<'PYSPLIT'
import ast
import json
import re
import sys
from pathlib import Path

import yaml

tasks_dir = Path(sys.argv[1])
task_set = sys.argv[2]
instructions_out = Path(sys.argv[3])
split_out = Path(sys.argv[4])
selected_out = Path(sys.argv[5])
task_list_file = sys.argv[6] if len(sys.argv) > 6 else ""

# Disclosed exclusions: these upstream tasks render their instructions
# dynamically (multiple templates, f-strings, or <label> placeholders), so
# a simulator-free manifest cannot freeze their final text.  The scan below
# re-derives this set mechanically and refuses to run if the checkout
# disagrees in either direction.
EXPECTED_UNIVERSE_COUNT = 54
EXPECTED_EXCLUDED = (
    "classify_objects_by_language",
    "general_pickup",
    "pour_by_language",
    "stack_blocks_by_language",
)
PLACEHOLDER = re.compile(r"<[^<>]+>")


def single_template(source):
    """Return the task's single verbatim template literal, or None."""

    defs = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "gen_instruction"
    ]
    if len(defs) != 1:
        return None
    func = defs[0]
    returns = [node for node in ast.walk(func) if isinstance(node, ast.Return)]
    if len(returns) != 1:
        return None
    value = returns[0].value
    if isinstance(value, ast.Name):
        assigns = [
            node.value
            for node in func.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == value.id
        ]
        if len(assigns) != 1:
            return None
        value = assigns[0]
    if not (isinstance(value, ast.List) and len(value.elts) == 1):
        return None
    element = value.elts[0]
    if not (
        isinstance(element, ast.Constant) and isinstance(element.value, str)
    ):
        return None
    text = element.value
    # Byte parity with the live DescManager rendering: the extracted literal
    # must appear verbatim in the task source, exactly like the single-task
    # grep -F gate.
    if not text.strip() or PLACEHOLDER.search(text) or text not in source:
        return None
    return text


names = sorted(path.stem for path in tasks_dir.glob("*.py"))
if len(names) != EXPECTED_UNIVERSE_COUNT:
    sys.exit(
        f"task universe drifted: found {len(names)} task sources, "
        f"expected {EXPECTED_UNIVERSE_COUNT}"
    )
templates = {}
excluded = []
for name in names:
    source = (tasks_dir / f"{name}.py").read_text(encoding="utf-8")
    text = single_template(source)
    if text is None:
        excluded.append(name)
    else:
        templates[name] = text
if tuple(excluded) != EXPECTED_EXCLUDED:
    sys.exit(
        "single-template exclusion set drifted from the disclosed set: "
        f"derived {excluded}, expected {list(EXPECTED_EXCLUDED)}"
    )
benchmark_tasks = [name for name in names if name in templates]
half = (len(benchmark_tasks) + 1) // 2
halves = {"A": benchmark_tasks[:half], "B": benchmark_tasks[half:]}
if task_list_file:
    requested = []
    for line in Path(task_list_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requested.append(line)
    if not requested:
        sys.exit("task list file selected no tasks")
    duplicates = sorted({t for t in requested if requested.count(t) > 1})
    if duplicates:
        sys.exit(f"task list file repeats tasks: {duplicates}")
    unknown = sorted(set(requested) - set(names))
    if unknown:
        sys.exit(f"task list names unknown tasks: {unknown}")
    unfreezable = sorted(set(requested) & set(excluded))
    if unfreezable:
        sys.exit(
            "task list includes tasks whose instruction cannot be frozen "
            f"(no single placeholder-free template): {unfreezable}"
        )
    # The served checkpoint family drives the dual_x5 embodiment only; a
    # task pinned to another robot_config would pass every file gate and
    # then die at the bridge's live-embodiment validation minutes into
    # Isaac Sim.  Fail closed here instead.
    task_meta = yaml.safe_load(
        (tasks_dir.parent / "config" / "_task.yml").read_text(encoding="utf-8")
    )
    common_robot = (task_meta.get("common") or {}).get(
        "robot_config", "dual_x5"
    )
    per_task_meta = task_meta.get("tasks") or {}
    wrong_embodiment = sorted(
        t
        for t in requested
        if ((per_task_meta.get(t) or {}).get("robot_config", common_robot))
        != "dual_x5"
    )
    if wrong_embodiment:
        sys.exit(
            f"task list includes non-dual_x5 embodiment tasks: "
            f"{wrong_embodiment}"
        )
    selected = sorted(set(requested))
else:
    selected = halves[task_set]

with instructions_out.open("x", encoding="utf-8") as stream:
    json.dump(
        {name: templates[name] for name in selected},
        stream,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    stream.write("\n")
with split_out.open("x", encoding="utf-8") as stream:
    json.dump(
        {
            "excluded_tasks": excluded,
            "selected_tasks": selected,
            "task_set": task_set,
            "task_set_a": halves["A"],
            "task_set_b": halves["B"],
            "task_universe": names,
        },
        stream,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    stream.write("\n")
with selected_out.open("x", encoding="utf-8") as stream:
    stream.write("".join(f"{name}\n" for name in selected))
print(
    f"task_set={task_set} selected={len(selected)} "
    f"benchmark_tasks={len(benchmark_tasks)} universe={len(names)}"
)
PYSPLIT

  TASK_ARGS=()
  while IFS= read -r task_name; do
    if [[ ! -f "${ROBODOJO_CHECKOUT}/task/RoboDojo/config/${task_name}.yml" ]]; then
      echo "required runtime input is missing: ${ROBODOJO_CHECKOUT}/task/RoboDojo/config/${task_name}.yml" >&2
      exit 66
    fi
    TASK_ARGS+=(--task "${task_name}")
  done <"${SELECTED_TASKS_FILE}"
  N_SELECTED_TASKS=$(( $(wc -l <"${SELECTED_TASKS_FILE}") ))

  timeout --signal=TERM --kill-after=30s "${PREFLIGHT_WATCHDOG_SECONDS}" \
    env PYTHONNOUSERSITE=1 PYTHONPATH="${SANA_REPO}" \
    "${SANA_PY}" -m \
    dev.rwm.simulation_evaluation.robodojo.common.preflight_manifest \
    --assets-root "${ROBODOJO_CHECKOUT}/Assets" \
    --config-name "${CONFIG_NAME}" \
    --eval-seed "${EVAL_SEED}" \
    "${TASK_ARGS[@]}" \
    --episodes-per-task "${EPISODES_PER_TASK}" \
    --task-set "${TASK_SET}" \
    --plan-output "${BENCHMARK_PLAN}" \
    --instructions "${INSTRUCTIONS_FILE}" \
    --output "${MANIFEST}" \
    --policy-session-id "${RUN_ID}" \
    >"${EVIDENCE_DIR}/manifest_preflight.stdout.log" \
    2>"${EVIDENCE_DIR}/manifest_preflight.stderr.log"
fi

write_run_status() {
  # write_run_status ENDPOINT_READY SERVER_RC HANDOFF_PATH FORCED
  "${SANA_PY}" - "${EVIDENCE_DIR}/run_status.json" \
    "$1" "$2" "$3" "$4" \
    "${CHECKPOINT}" "${STATUS_TASK}" "${NODE_FQDN}" "${RUN_ID}" \
    "${SANA_RPC_BIND_HOST}" "${SANA_RPC_ADVERTISE_HOST}" \
    "${SANA_RPC_PORT}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
endpoint_ready = sys.argv[2] == "true"
server_rc = int(sys.argv[3]) if sys.argv[3] != "" else None
handoff = sys.argv[4] or None
forced = sys.argv[5] == "true"
value = {
    "advertise_host": sys.argv[11],
    "bind_host": sys.argv[10],
    "checkpoint": sys.argv[6],
    "client_handoff": handoff,
    # The simulator client runs on a remote machine; its verdict lives with
    # that machine's evidence, never in this file.
    "client_verdict": None,
    "compute_node": sys.argv[8],
    "endpoint_ready": endpoint_ready,
    "kind": "robodojo_server_only",
    "rpc_port": int(sys.argv[12]),
    "run_id": sys.argv[9],
    "server_forced_shutdown": forced,
    "server_rc": server_rc,
    "task": sys.argv[7],
}
with output.open("x", encoding="utf-8") as stream:
    json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
    stream.write("\n")
PY
}

SERVER_PID=
SERVER_RUNTIME_PID=
SERVER_RC=
SERVER_FORCED_SHUTDOWN=false
SHUTDOWN_REQUESTED=false
request_server_shutdown() {
  if [[ -n "${SERVER_RUNTIME_PID}" ]] \
      && kill -0 "${SERVER_RUNTIME_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_RUNTIME_PID}" 2>/dev/null || true
  elif [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
  fi
}
stop_server() {
  local waited_seconds=0
  if [[ -z "${SERVER_PID}" ]]; then
    return 0
  fi
  request_server_shutdown
  while kill -0 "${SERVER_PID}" 2>/dev/null; do
    if ((waited_seconds >= SERVER_SHUTDOWN_GRACE_SECONDS)); then
      SERVER_FORCED_SHUTDOWN=true
      if [[ -n "${SERVER_RUNTIME_PID}" ]]; then
        kill -KILL "${SERVER_RUNTIME_PID}" 2>/dev/null || true
      fi
      kill -KILL "${SERVER_PID}" 2>/dev/null || true
      break
    fi
    sleep 1
    ((waited_seconds += 1))
  done
  if wait "${SERVER_PID}" 2>/dev/null; then
    SERVER_RC=0
  else
    SERVER_RC=$?
  fi
  SERVER_PID=
  SERVER_RUNTIME_PID=
}
handle_stop_signal() {
  # A trapped TERM/INT (operator kill or Ctrl-C) turns into one
  # graceful server shutdown request; the main wait loop below finishes the
  # grace/KILL sequence.
  SHUTDOWN_REQUESTED=true
  request_server_shutdown
}
cleanup_all() {
  stop_server || true
}
trap handle_stop_signal TERM INT
trap cleanup_all EXIT

if [[ "${SANA_DRY_RUN:-0}" == 1 ]]; then
  echo "SANA_DRY_RUN=1: preflight gates and manifest passed; not starting the server (evidence=${EVIDENCE_DIR})"
  trap - TERM INT EXIT
  exit 0
fi

# The server owns GPU ${SANA_SERVER_GPU} of this node.  --eager-load
# makes the model load part of server startup, so a loading failure kills
# the run before any handoff is published and the startup receipt records
# model_load_state="loaded".
# gemma-2-2b-it-local is a REPO-RELATIVE path (diffusion/model/builder.py); the
# sbatch relied on being submitted from the repo root.  Every path above was
# resolved against the invocation directory, so the cd happens only now.
cd -- "${SANA_REPO}"
env CUDA_VISIBLE_DEVICES="${SANA_SERVER_GPU}" \
  SANA_INFERENCE_WORK_DIR="${EVIDENCE_DIR}/model_work_dir" \
  PYTHONNOUSERSITE=1 PYTHONPATH="${SANA_REPO}" PYTHONUNBUFFERED=1 \
  WANDB_MODE=disabled \
  "${SANA_PY}" -m \
  dev.rwm.simulation_evaluation.robodojo.server.main serve \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --vae-pretrained "${VAE_PRETRAINED}" \
  --normalization "${NORMALIZATION}" \
  --training-log "${FROZEN_TRAINING_LOG}" \
  --manifest "${MANIFEST}" \
  --inference-num-frames "${INFERENCE_NUM_FRAMES}" \
  --n-action-steps "${N_ACTION_STEPS}" \
  --control-mode joint \
  --joint-limit-policy clip_to_smoke_envelope \
  --runtime-sana-commit "${RUNTIME_COMMIT}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --run-id "${RUN_ID}" \
  --rpc-host "${SANA_RPC_ADVERTISE_HOST}" \
  --rpc-port "${SANA_RPC_PORT}" \
  --bind-host "${SANA_RPC_BIND_HOST}" \
  --device cuda \
  --eager-load \
  "${PREDICTION_VIDEO_ARGS[@]}" \
  >"${EVIDENCE_DIR}/server.stdout.log" \
  2>"${EVIDENCE_DIR}/server.stderr.log" &
SERVER_PID=$!

for ((waited_seconds = 0; waited_seconds < SERVER_STARTUP_TIMEOUT_SECONDS; waited_seconds++)); do
  if [[ -s "${ARTIFACT_DIR}/resolved_endpoint.json" \
      && -s "${ARTIFACT_DIR}/server_startup_receipt.json" \
      && -s "${ARTIFACT_DIR}/session_config.json" ]]; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    set +e
    wait "${SERVER_PID}"
    early_server_rc=$?
    set -e
    SERVER_PID=
    printf '%s\n' "${early_server_rc}" >"${EVIDENCE_DIR}/server_exit_code.txt"
    write_run_status false "${early_server_rc}" "" false
    echo "learned server exited before publishing its endpoint" >&2
    exit 70
  fi
  sleep 1
done
if [[ ! -s "${ARTIFACT_DIR}/resolved_endpoint.json" \
    || ! -s "${ARTIFACT_DIR}/server_startup_receipt.json" \
    || ! -s "${ARTIFACT_DIR}/session_config.json" ]]; then
  stop_server
  trap - EXIT
  printf '%s\n' "${SERVER_RC}" >"${EVIDENCE_DIR}/server_exit_code.txt"
  write_run_status false "${SERVER_RC}" "" "${SERVER_FORCED_SHUTDOWN}"
  echo "timed out waiting for the learned RoboDojo endpoint" >&2
  exit 70
fi

SERVER_RUNTIME_PID=$("${SANA_PY}" -c \
  'import json,sys; v=json.load(open(sys.argv[1], encoding="utf-8"))["server_pid"]; (isinstance(v,int) and not isinstance(v,bool) and v>0) or sys.exit("invalid server_pid"); print(v)' \
  "${ARTIFACT_DIR}/resolved_endpoint.json")
if ! kill -0 "${SERVER_RUNTIME_PID}" 2>/dev/null; then
  echo "published server PID is not alive: ${SERVER_RUNTIME_PID}" >&2
  exit 70
fi

# The handoff is only honest for a listening, eager-loaded endpoint that
# advertises exactly the host/port the tunnel instructions name; gate all of
# that before publishing anything to the client machine.
set +e
"${SANA_PY}" - "${ARTIFACT_DIR}/resolved_endpoint.json" \
  "${ARTIFACT_DIR}/server_startup_receipt.json" \
  "${SANA_RPC_ADVERTISE_HOST}" "${SANA_RPC_PORT}" <<'PY'
import json
import sys
from pathlib import Path

endpoint = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
startup = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
advertise_host, rpc_port = sys.argv[3], int(sys.argv[4])

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(endpoint["model_load_state"] == "loaded", "endpoint model load state")
require(startup["model_load_state"] == "loaded", "startup model load state")
require(startup["status"] == "listening", "startup status")
require(startup["server_pid"] == endpoint["server_pid"], "startup PID")
require(
    endpoint["endpoint"] == {"host": advertise_host, "port": rpc_port},
    "published endpoint host/port disagrees with the launch knobs",
)
PY
ENDPOINT_GATE_RC=$?
set -e
if [[ "${ENDPOINT_GATE_RC}" -ne 0 ]]; then
  stop_server
  trap - EXIT
  printf '%s\n' "${SERVER_RC}" >"${EVIDENCE_DIR}/server_exit_code.txt"
  write_run_status false "${SERVER_RC}" "" "${SERVER_FORCED_SHUTDOWN}"
  echo "published endpoint failed the loaded-and-listening gate" >&2
  exit 70
fi

SESSION_HEX=$(sha256sum "${ARTIFACT_DIR}/session_config.json")
SESSION_HEX=${SESSION_HEX%% *}
SESSION_SHA=sha256:${SESSION_HEX}
printf '%s\n' "${SESSION_SHA}" >"${EVIDENCE_DIR}/session_config.sha256"

# Client handoff bundle: everything the remote simulator machine needs to
# join this session.  The session config copy is byte-verified against the
# server-published original so the recorded sha256 provably names the copied
# bytes.
mkdir -p -- "${HANDOFF_DIR}"
cp --preserve=timestamps -- \
  "${ARTIFACT_DIR}/session_config.json" "${HANDOFF_DIR}/session_config.json"
if ! cmp -s -- "${ARTIFACT_DIR}/session_config.json" \
    "${HANDOFF_DIR}/session_config.json"; then
  echo "handoff session_config.json copy does not match the published original" >&2
  exit 70
fi
printf '%s\n' "${SESSION_SHA}" >"${HANDOFF_DIR}/session_config.sha256"

# The session config binds the episode manifest by an absolute cluster
# path; a split-deployment client has no /lustre, so the loader falls back
# to the manifest's basename next to session_config.json (digest-checked
# either way).  Ship those bytes in the bundle.
cp --preserve=timestamps -- \
  "${MANIFEST}" "${HANDOFF_DIR}/preflight_manifest.json"
if ! cmp -s -- "${MANIFEST}" "${HANDOFF_DIR}/preflight_manifest.json"; then
  echo "handoff preflight_manifest.json copy does not match the original" >&2
  exit 70
fi

# The task-set lane additionally ships the ordered replay plan the client
# runner mirrors verbatim.
if [[ -n "${TASK_SET}" ]]; then
  cp --preserve=timestamps -- \
    "${BENCHMARK_PLAN}" "${HANDOFF_DIR}/benchmark_plan.json"
  if ! cmp -s -- "${BENCHMARK_PLAN}" "${HANDOFF_DIR}/benchmark_plan.json"; then
    echo "handoff benchmark_plan.json copy does not match the original" >&2
    exit 70
  fi
fi

if [[ -z "${TASK_SET}" ]]; then
cat >"${HANDOFF_DIR}/CLIENT_README.txt" <<EOF
RoboDojo split evaluation: client handoff
=========================================

Server endpoint (cluster side)
  compute node    : ${NODE_FQDN}
  launcher pid    : $$ (also in ${EVIDENCE_DIR}/launcher.pid)
  rpc port        : ${SANA_RPC_PORT}
  bind host       : ${SANA_RPC_BIND_HOST}
  advertised host : ${SANA_RPC_ADVERTISE_HOST}
  run id          : ${RUN_ID}
  sana commit     : ${RUNTIME_COMMIT}
  task / seed     : ${TASK} / ${EVAL_SEED}
  session sha256  : ${SESSION_SHA}

The frozen session config advertises rpc host ${SANA_RPC_ADVERTISE_HOST}, so
the simulator client dials ${SANA_RPC_ADVERTISE_HOST}:${SANA_RPC_PORT} on the
CLIENT machine; the SSH tunnel below carries that loopback port to this
compute node.  The endpoint lives only as long as this launcher process.

1) If the client runs on ANOTHER node, open this SSH tunnel from the client
   machine and keep it running (skip it when the client runs on this very
   node: it dials 127.0.0.1:${SANA_RPC_PORT} directly):

   ssh -N -L ${SANA_RPC_PORT}:127.0.0.1:${SANA_RPC_PORT} $(id -un)@${NODE_FQDN}

2) The bundle lives at ${HANDOFF_DIR}; on the shared home every lN node
   already sees it, so copy only when the client machine is elsewhere:

   scp -r $(id -un)@${NODE_FQDN}:${HANDOFF_DIR} ./client_handoff

3) On the client machine, from a Sana-private checkout at the sana commit
   above, with the tunnel from step 1 still open:

   bash ${REMOTE_CLIENT_SCRIPT} ./client_handoff

Bundle contents
  session_config.json      frozen session config published by the live server
  session_config.sha256    sha256:<hex> of session_config.json, byte-verified
  preflight_manifest.json  episode manifest bound by the session config; the
                           client loader resolves it next to
                           session_config.json when the cluster-absolute
                           path does not exist (digest-checked either way)
  CLIENT_README.txt        this file

To end the session cleanly (from any lN node; it signals the launcher on ${NODE_FQDN}):

   ssh $(id -un)@${NODE_FQDN} bash ${LAUNCHER_DIR}/stop_server.sh ${EVIDENCE_DIR}
   # equivalent, ON ${NODE_FQDN} itself: kill -TERM $$   (pid also in ${EVIDENCE_DIR}/launcher.pid)
EOF
else
cat >"${HANDOFF_DIR}/CLIENT_README.txt" <<EOF
RoboDojo split evaluation: client handoff (task-set benchmark)
==============================================================

Server endpoint (cluster side)
  compute node    : ${NODE_FQDN}
  launcher pid    : $$ (also in ${EVIDENCE_DIR}/launcher.pid)
  rpc port        : ${SANA_RPC_PORT}
  bind host       : ${SANA_RPC_BIND_HOST}
  advertised host : ${SANA_RPC_ADVERTISE_HOST}
  run id          : ${RUN_ID}
  sana commit     : ${RUNTIME_COMMIT}
  task set / seed : ${TASK_SET} (${N_SELECTED_TASKS} tasks x ${EPISODES_PER_TASK} episodes) / ${EVAL_SEED}
  session sha256  : ${SESSION_SHA}

The frozen session config advertises rpc host ${SANA_RPC_ADVERTISE_HOST}, so
the simulator client dials ${SANA_RPC_ADVERTISE_HOST}:${SANA_RPC_PORT} on the
CLIENT machine; the SSH tunnel below carries that loopback port to this
compute node.  The endpoint lives only as long as this launcher process.

1) If the client runs on ANOTHER node, open this SSH tunnel from the client
   machine and keep it running (skip it when the client runs on this very
   node: it dials 127.0.0.1:${SANA_RPC_PORT} directly):

   ssh -N -L ${SANA_RPC_PORT}:127.0.0.1:${SANA_RPC_PORT} $(id -un)@${NODE_FQDN}

2) The bundle lives at ${HANDOFF_DIR}; on the shared home every lN node
   already sees it, so copy only when the client machine is elsewhere:

   scp -r $(id -un)@${NODE_FQDN}:${HANDOFF_DIR} ./client_handoff

3) On the client machine, from a Sana-private checkout at the sana commit
   above, with the tunnel from step 1 still open, run this task set's half
   of the benchmark:

   bash ${REMOTE_BENCHMARK_CLIENT_SCRIPT} ./client_handoff

   The runner replays benchmark_plan.json verbatim: every task in plan
   order, the plan's layout ids in order, one server session shared by all
   episodes.  The other task set runs against its own server job from its
   own handoff bundle.

Bundle contents
  session_config.json      frozen session config published by the live server
  session_config.sha256    sha256:<hex> of session_config.json, byte-verified
  preflight_manifest.json  episode manifest bound by the session config; the
                           client loader resolves it next to
                           session_config.json when the cluster-absolute
                           path does not exist (digest-checked either way)
  benchmark_plan.json      ordered replay plan for this task set:
                           episodes_per_task, eval_seed, task_set, and one
                           record per task carrying task, instruction,
                           start_ordinal, layout_ids
  CLIENT_README.txt        this file

To end the session cleanly (from any lN node; it signals the launcher on ${NODE_FQDN}):

   ssh $(id -un)@${NODE_FQDN} bash ${LAUNCHER_DIR}/stop_server.sh ${EVIDENCE_DIR}
   # equivalent, ON ${NODE_FQDN} itself: kill -TERM $$   (pid also in ${EVIDENCE_DIR}/launcher.pid)
EOF
fi

echo "===== CLIENT HANDOFF ====="
cat "${HANDOFF_DIR}/CLIENT_README.txt"
echo "===== END CLIENT HANDOFF ====="
echo "client handoff bundle: ${HANDOFF_DIR}"

# This process IS the server from here on: stay alive until the server
# exits on its own or an operator sends TERM/INT.
# The poll bounds trap latency; stop_server owns the grace/KILL sequence and
# reaps the real exit status in every path.
while kill -0 "${SERVER_PID}" 2>/dev/null; do
  if [[ "${SHUTDOWN_REQUESTED}" == true ]]; then
    break
  fi
  sleep 5
done
stop_server
trap - TERM INT EXIT

printf '%s\n' "${SERVER_RC}" >"${EVIDENCE_DIR}/server_exit_code.txt"
write_run_status true "${SERVER_RC}" "${HANDOFF_DIR}" "${SERVER_FORCED_SHUTDOWN}"

find "${EVIDENCE_DIR}" -type f \
  ! -name 'slurm-*.out' ! -name 'slurm-*.err' \
  ! -name checksums.sha256 ! -name checksums.verify.log \
  -print0 | sort -z | xargs -0 sha256sum \
  >"${EVIDENCE_DIR}/checksums.sha256"
sha256sum -c "${EVIDENCE_DIR}/checksums.sha256" \
  >"${EVIDENCE_DIR}/checksums.verify.log"

if [[ -z "${SERVER_RC}" || "${SERVER_RC}" -ne 0 \
    || "${SERVER_FORCED_SHUTDOWN}" == true ]]; then
  echo "robodojo_server_only FAILED; inspect ${EVIDENCE_DIR}/run_status.json" >&2
  exit 1
fi

echo "robodojo_server_only PASS: task=${STATUS_TASK} endpoint=${SANA_RPC_ADVERTISE_HOST}:${SANA_RPC_PORT} evidence=${EVIDENCE_DIR}"
