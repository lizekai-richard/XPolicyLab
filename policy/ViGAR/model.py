"""ViGAR adapter: a subgoal planner and a world-action policy for RoboTwin 2.0.

The planner and the policy need different Cosmos runtimes, so each runs in the
model server that ships with ViGAR. The adapter starts both servers and reuses
ViGAR's RoboTwin client code for subgoal generation, state encoding and action
decoding, so the models receive the same requests as in ViGAR's own evaluation.
"""

import atexit
import ctypes
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

POLICY_DIR = Path(__file__).resolve().parent
# XPolicyLab camera names -> RoboTwin camera names expected by ViGAR.
CAMERAS = {"cam_head": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}
# OpenCV camera axes (x right, y down, z forward) -> OpenGL axes (x right, y up, z backward).
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
# ViGAR decodes a qpos chunk as [left arm, left gripper, right arm, right gripper].
QPOS_WIDTH = 14
ACTION_HORIZON, STATE_DIM = 48, 49


def _resolve_path(value):
    path = Path(os.path.expanduser(str(value)))
    return path if path.is_absolute() else (POLICY_DIR / path).resolve()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_server_session():
    # Own process group for cleanup; terminate the server if this process dies.
    os.setsid()
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _action_slices(env_cfg_type):
    dims = get_robot_action_dim_info(env_cfg_type)
    if len(dims["arm_dim"]) != 2:
        raise ValueError("ViGAR controls a dual-arm robot")
    slices, start = {}, 0
    for side, arm, ee in zip(("left", "right"), dims["arm_dim"], dims["ee_dim"]):
        slices[f"{side}_arm_joint_state"] = slice(start, start + arm)
        slices[f"{side}_ee_joint_state"] = slice(start + arm, start + arm + ee)
        start += arm + ee
    if start != QPOS_WIDTH:
        raise ValueError(f"{env_cfg_type} has {start} joint actions; ViGAR outputs {QPOS_WIDTH} (6+1 per arm)")
    return slices


def _server_env(base, gpu, pythonpath, master_port, run_dir):
    """Single-process environment for one ViGAR model server, as in ViGAR's evaluation."""
    env = dict(
        base,
        CUDA_VISIBLE_DEVICES=str(gpu),
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
        LOCAL_WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
        PYTHONPATH=":".join(map(str, pythonpath)),
        WANDB_MODE="disabled",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONHASHSEED="42",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        FLASH_ATTENTION_DETERMINISTIC="1",
        COSMOS_TRAINING="1",
        TOKENIZERS_PARALLELISM="false",
        IMAGINAIRE_OUTPUT_ROOT=str(run_dir / "model_outputs"),
    )
    env.pop("WANDB_API_KEY", None)
    return env


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        if model_cfg["action_type"] != "joint":
            raise ValueError("The released ViGAR checkpoint is evaluated with action_type=joint")
        self.action_slices = _action_slices(model_cfg["env_cfg_type"])
        self.task_name = model_cfg.get("task_name")
        self.replan_steps = int(model_cfg.get("replan_steps", 32))
        if not 1 <= self.replan_steps <= ACTION_HORIZON:
            raise ValueError(f"replan_steps must be in [1, {ACTION_HORIZON}]")
        self.generation_seed = int(model_cfg.get("generation_seed", 9000))
        self.sampling = dict(model_cfg.get("policy_sampling") or dict(num_steps=10, guidance=1.0, shift=2.0))

        self.vigar_root = _resolve_path(model_cfg.get("vigar_root") or "source_vigar")
        bundle = resolve_checkpoint_root(model_cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR)
        self.policy_checkpoint = (bundle / "weights/policy/robotwin_c2r").resolve()
        self.planner_checkpoint = (bundle / "weights/subgoal_planner/robotwin_subgoal_planner").resolve()
        for path in (self.vigar_root / "evaluation", self.policy_checkpoint, self.planner_checkpoint):
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; see policy/ViGAR/README.md")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = _resolve_path(model_cfg.get("log_dir") or "logs") / f"{self.task_name}-{stamp}-{os.getpid()}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.processes, self.streams = [], []
        self.planner = self.producer = None
        atexit.register(self.close)
        signal.signal(signal.SIGTERM, self._on_sigterm)
        self._start_servers()
        self._connect()

        self.case = {}
        self.episodes = 0
        self.episode = None
        self.obs = None
        print(f"[ViGAR] planner and policy ready; logs in {self.run_dir}")

    # ------------------------------------------------------------------ servers
    def _vigar_env(self, python):
        # Environment of ViGAR's scripts/env.sh, without reading a local .env file.
        output = subprocess.run(
            ["bash", "-c", 'source "$1" >/dev/null && env -0', "bash", str(self.vigar_root / "scripts/env.sh")],
            env=dict(os.environ, PYTHON_BIN=str(python), VIGAR_ENV_FILE=os.devnull),
            check=True,
            capture_output=True,
        ).stdout.decode()
        env = dict(item.split("=", 1) for item in output.split("\0") if "=" in item)
        env.update(
            WAN_VAE_PATH=str(self.policy_checkpoint / "vae/Wan2.2_VAE.pth"),
            QWEN_TOKENIZER_PATH=str(self.policy_checkpoint / "text_tokenizer"),
        )
        return env

    def _spawn(self, command, log_name, env, cwd):
        stream = (self.run_dir / log_name).open("w")
        self.streams.append(stream)
        process = subprocess.Popen(
            [str(part) for part in command],
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(cwd),
            preexec_fn=_start_server_session,
        )
        self.processes.append(process)
        return process

    def _start_servers(self):
        python = _resolve_path(self.model_cfg.get("vigar_python") or self.vigar_root / ".venv-model/bin/python")
        base = self._vigar_env(python)
        # One GPU id: planner and policy share it. Two ids: planner on the first, policy on the second.
        # planner_gpu_id in deploy.yml moves the planner to another physical GPU.
        gpus = [gpu.strip() for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if gpu.strip()]
        planner_gpu, policy_gpu = gpus[0], gpus[-1]
        if self.model_cfg.get("planner_gpu_id") is not None:
            planner_gpu = str(self.model_cfg["planner_gpu_id"])
        self.planner_port, self.policy_port, planner_master, policy_master = (_free_port() for _ in range(4))

        evaluation = self.vigar_root / "evaluation"
        service = evaluation / "planner_service"
        release = self.vigar_root / "subgoal_planner/inference_runtime"
        source = self.vigar_root / "vigar/source"
        planner_env = _server_env(base, planner_gpu, [release, evaluation, service], planner_master, self.run_dir)
        policy_env = _server_env(
            base, policy_gpu, [source, source / "third_party/cosmos_runtime"], policy_master, self.run_dir
        )
        if self.model_cfg.get("fast_inference"):
            planner_env["VIGAR_PLANNER_BACKEND_CONFIG"] = str(self.vigar_root / "configs/planner_backend_fast.json")
            policy_env["VIGAR_FAST_INFERENCE"] = "1"

        planner = self._spawn(
            [python, service / "serve_planner.py", "--checkpoint", self.planner_checkpoint,
             "--sft-toml", release / "episode_image_edit_nano.toml", "--port", self.planner_port],
            "planner.log", planner_env, release,
        )
        policy = self._spawn(
            [python, "-m", "recipes.simulation.robotwin.vigar.server", "--checkpoint", self.policy_checkpoint,
             "--weights", "ema", "--host", "127.0.0.1", "--port", self.policy_port,
             "--output", self.run_dir / "policy_outputs"],
            "policy.log", policy_env, source,
        )

        sys.path.append(str(service))
        from robotwin_wire import Client

        deadline = time.monotonic() + float(self.model_cfg.get("server_startup_timeout_s", 1800))
        while True:
            for process, log_name in ((planner, "planner.log"), (policy, "policy.log")):
                if process.poll() is not None:
                    raise RuntimeError(f"A ViGAR model server exited during startup; see {self.run_dir / log_name}")
            try:
                url = f"http://127.0.0.1:{self.policy_port}/health"
                with urllib.request.urlopen(url, timeout=2) as response:
                    response.read()
                client = Client(self.planner_port, timeout=3)
                try:
                    ping = client.call(dict(cmd="ping"))
                finally:
                    client.close()
                if ping["checkpoint"] != str(self.planner_checkpoint):
                    raise RuntimeError(f"Planner server loaded {ping['checkpoint']}")
                return
            except (OSError, ConnectionError, TimeoutError):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"ViGAR model servers did not start; see {self.run_dir}")
                time.sleep(3)

    def _connect(self):
        for path in ("evaluation", "evaluation/planner_service", "vigar/source"):
            if str(self.vigar_root / path) not in sys.path:
                sys.path.append(str(self.vigar_root / path))
        from async_subgoal import AsyncSubgoalProducer
        from native_async_goal import NativeAsyncGoalClient
        from recipes.simulation.robotwin.common.geometry import controller_actions, encode_observation
        from recipes.simulation.robotwin.common.transport import PolicyClient
        from robotwin_async_gateway import concat_views
        from robotwin_wire import Client
        from sync_subgoal import SyncSubgoalProducer

        self.encode_observation, self.controller_actions = encode_observation, controller_actions
        refresh = self.model_cfg.get("goal_refresh_mode", "sync_current")
        timeout = float(self.model_cfg.get("model_request_timeout_s", 600))
        self.planner = Client(self.planner_port, timeout=timeout)
        self.producer = (SyncSubgoalProducer if refresh == "sync_current" else AsyncSubgoalProducer)(self.planner)
        self.client = NativeAsyncGoalClient(
            PolicyClient(f"http://127.0.0.1:{self.policy_port}", timeout),
            self.producer,
            concat_views,
            self.run_dir / "subgoals.jsonl",
            policy_obs_order=self.model_cfg.get("input_color_order", "rgb"),
            goal_refresh_mode=refresh,
        )
        health = self.client.call("/health")
        if health["checkpoint"] != str(self.policy_checkpoint):
            raise RuntimeError(f"Policy server loaded {health['checkpoint']}")
        if (health.get("action_horizon"), health.get("action_dim")) != (ACTION_HORIZON, STATE_DIM):
            raise RuntimeError("The policy server does not serve 48-step, 49-D actions")
        if "qpos" not in health.get("control_modes", []):
            raise RuntimeError("The policy checkpoint does not supervise joint control")

    def close(self):
        if self.producer is not None:
            self.producer.stop()
            self.producer = None
        if self.planner is not None:
            self.planner.close()
            self.planner = None
        for process in reversed(self.processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in self.processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        self.processes = []
        for stream in self.streams:
            stream.close()
        self.streams = []

    def _on_sigterm(self, signum, frame):
        self.close()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    # ------------------------------------------------------------- observations
    @staticmethod
    def _robotwin_observation(obs):
        """Rebuild the RoboTwin observation fields that ViGAR's state encoder reads."""
        vision, state = obs["vision"], obs["state"]
        extrinsic = np.eye(4)
        extrinsic[:3, :] = np.asarray(vision["cam_head"]["extrinsics_matrix"], dtype=np.float64)[:3, :4]
        cameras = {name: {"rgb": np.asarray(vision[key]["color"])} for key, name in CAMERAS.items()}
        cameras["head_camera"]["cam2world_gl"] = np.linalg.inv(extrinsic) @ CV_TO_GL
        joint_action, endpose = {}, {}
        for side in ("left", "right"):
            gripper = float(np.asarray(state[f"{side}_ee_joint_state"]).reshape(-1)[0])
            pose = np.asarray(state[f"{side}_ee_pose"], dtype=np.float64).reshape(-1).copy()
            pose[3:] /= np.linalg.norm(pose[3:])  # poses arrive as float32
            joint_action[f"{side}_arm"] = np.asarray(state[f"{side}_arm_joint_state"], dtype=np.float64)
            joint_action[f"{side}_gripper"] = gripper
            endpose[f"{side}_endpose"] = pose
            endpose[f"{side}_gripper"] = gripper
        return {"observation": cameras, "joint_action": joint_action, "endpose": endpose}

    def prepare_case(self, case_meta=None):
        self.case = dict(case_meta or {})

    def reset(self):
        # RoboTwin sends the task and the episode seed through prepare_case; the
        # planner samples with the episode seed, as in ViGAR's own evaluation.
        task = self.case.get("task_name") or self.task_name
        seed = int(self.case.get("seed", self.episodes))
        self.episode = (str(task), seed)
        self.client.reset(task, seed)
        self.episodes += 1
        self.queries = 0
        self.observations = 0
        self.obs = None

    def update_obs(self, obs):
        self.obs = obs
        self.observations += 1

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError("ViGAR keeps one subgoal state per server; set eval_batch: false")
        self.update_obs(obs_list[0])

    def get_action(self):
        if self.episode is None or self.obs is None:
            raise RuntimeError("Call reset() and update_obs() before get_action()")
        instruction = self.obs.get("instruction") or (self.obs.get("instructions") or [""])[0]
        request = self.encode_observation(self._robotwin_observation(self.obs))
        request.update(
            instruction=instruction,
            generation_seed=self.generation_seed + self.queries,
            artifact_context=dict(
                task=self.episode[0],
                episode_seed=self.episode[1],
                query_index=self.queries,
                control_step=self.observations - 1,
            ),
            sampling=dict(self.sampling),
        )
        result = self.client.call("/infer", request)
        chunk = self.controller_actions(result["absolute"], result["valid"], request["cam2world_gl"], "qpos")
        self.queries += 1
        return [
            {key: np.asarray(row[part], dtype=np.float32) for key, part in self.action_slices.items()}
            for row in chunk[: self.replan_steps]
        ]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and len(env_idx_list) != 1:
            raise ValueError("ViGAR keeps one subgoal state per server; set eval_batch: false")
        return [self.get_action()]
