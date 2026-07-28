"""Vectorized self-play worker.

Each worker process runs `batch_games` episodes concurrently as generators.
Every round, it collects one pending network request per episode, evaluates
them as ONE batched forward pass, and feeds results back. Batching is
across games, so per-game search semantics are exactly the sequential ones.

Weights sync via the checkpoint file's mtime (the learner writes it
atomically). Study + resignation switch on when the gate file exists
(the learner creates it once the value head has matured)."""

import torch  # noqa: I001  (torch before numpy; see README)

import json
import os
import queue as queue_mod
from dataclasses import replace

import numpy as np

from .accel import autocast, maybe_compile, setup_perf, to_np  # noqa: F401
from .encoding import FEATURES, NUM_ACTIONS
from .model import ModelConfig, PolicyQValueNet, extract_state
from .schedule import q_trust_at, study_config_at
from .search import SearchConfig
from .selfplay import SelfPlayConfig, play_game_gen
from .study import StudyConfig, study_game_gen

RELOAD_EVERY_ROUNDS = 32


def episode_gen(scfg, spcfg, stcfg, rng, gate_fn, resign_gate_fn=None):
    """One full episode: a game, plus study if the study gate is open.
    Resignation gates separately (resign_gate_fn) so it can switch on EARLIER
    than study: resignation needs only a usable value head, study needs a
    matured Q-head. Defaults to the study gate when no separate gate is given."""
    if resign_gate_fn is None:
        resign_gate_fn = gate_fn
    record = yield from play_game_gen(scfg, spcfg, rng, resign_enabled=resign_gate_fn())
    if stcfg is not None and gate_fn():
        extra = yield from study_game_gen(record, scfg, stcfg, rng)
        record.samples.extend(extra)
    return record


def run_vector_selfplay(
    model,
    device,
    scfg: SearchConfig,
    spcfg: SelfPlayConfig,
    stcfg: StudyConfig | None,
    gate_fn,
    batch_games: int,
    master_rng: np.random.Generator,
    on_record,
    should_stop,
    on_round=None,
    cfg_fn=None,
    resign_gate_fn=None,
):
    """Core driver loop. Calls on_record(record) for each finished episode;
    runs until should_stop() is true. cfg_fn(), if given, returns the
    (scfg, stcfg) to use for each new episode (game-count curricula)."""

    def new_episode():
        rng = np.random.default_rng(int(master_rng.integers(2**63)))
        s_cfg, st_cfg = cfg_fn() if cfg_fn is not None else (scfg, stcfg)
        gen = episode_gen(s_cfg, spcfg, st_cfg, rng, gate_fn, resign_gate_fn)
        return gen, gen.send(None)

    gens, pending = [], []
    for _ in range(batch_games):
        g, x = new_episode()
        gens.append(g)
        pending.append(x)

    # Persistent host+device staging buffers: each round restacks the pending
    # leaf features into the same host array and copies once to the GPU, so the
    # hot loop allocates nothing per round. Reuse is safe because the packed
    # D2H below fully syncs the round before the next stack overwrites cpu_stage.
    na = NUM_ACTIONS
    cpu_stage = np.empty((batch_games, 64, FEATURES), dtype=np.float32)
    dev_stage = torch.empty((batch_games, 64, FEATURES), device=device)

    rounds = 0
    while not should_stop():
        if on_round is not None and rounds % RELOAD_EVERY_ROUNDS == 0:
            on_round(rounds)
        np.stack(pending, out=cpu_stage)
        dev_stage.copy_(torch.from_numpy(cpu_stage), non_blocking=True)
        with torch.inference_mode(), autocast(device):
            logits, q, v = model(dev_stage)
        # ONE packed GPU->CPU transfer/sync instead of three separate .cpu()
        # round-trips: cat [B,4096]|[B,4096]|[B,1] -> [B,8193], slice on host.
        arr = torch.cat([logits, q, v[:, None]], dim=1).float().cpu().numpy()
        for i in range(len(gens)):
            try:
                pending[i] = gens[i].send((arr[i, :na], arr[i, na : 2 * na], arr[i, 2 * na]))
            except StopIteration as e:
                if not on_record(e.value):
                    return
                gens[i], pending[i] = new_episode()
        rounds += 1


def vector_worker(
    worker_id: int,
    ckpt_path: str,
    gate_path: str,
    out_queue,
    stop_event,
    search_kwargs: dict,
    selfplay_kwargs: dict,
    study_kwargs: dict | None,
    model_kwargs: dict,
    batch_games: int = 48,
    threads: int = 2,
    device_str: str = "cpu",
    progress_path: str | None = None,
    compile_model: bool = False,
    resign_gate_path: str | None = None,
):
    torch.set_num_threads(threads)
    # In a hybrid CPU+MPS layout, nice the CPU workers down so their heavy CPU
    # forwards don't preempt the MPS workers' latency-sensitive single-threaded
    # tree-ops (which is what actually feeds the GPU).
    if str(device_str).startswith("cpu"):
        try:
            os.nice(10)
        except OSError:
            pass
    setup_perf(device_str)
    device = torch.device(device_str)
    model = PolicyQValueNet(ModelConfig(**model_kwargs)).to(device)
    model.eval()
    model = maybe_compile(model, device, compile_model)
    master_rng = np.random.default_rng([worker_id, os.getpid()])
    scfg = SearchConfig(**search_kwargs)
    spcfg = SelfPlayConfig(**selfplay_kwargs)
    stcfg = StudyConfig(**study_kwargs) if study_kwargs is not None else None

    state = {"mtime": 0.0, "games": 0}

    def reload_ckpt(_rounds):
        try:
            m = os.path.getmtime(ckpt_path)
            if m != state["mtime"]:
                sd = extract_state(ckpt_path)
                model.load_state_dict(sd)
                state["mtime"] = m
        except (OSError, RuntimeError, KeyError):
            pass  # mid-write or missing; retry next round
        if progress_path is not None:
            try:
                with open(progress_path) as f:
                    state["games"] = int(json.load(f).get("games", 0))
            except (OSError, ValueError):
                pass  # mid-write or missing; keep last known count

    def cfg_fn():
        games = state["games"]
        s_cfg = replace(scfg, q_trust=q_trust_at(games))
        st_cfg = study_config_at(games, stcfg) if stcfg is not None else None
        return s_cfg, st_cfg

    def on_record(record) -> bool:
        while not stop_event.is_set():
            try:
                out_queue.put(record, timeout=1.0)
                return True
            except queue_mod.Full:
                continue
        return False

    run_vector_selfplay(
        model,
        device,
        scfg,
        spcfg,
        stcfg,
        gate_fn=lambda: os.path.exists(gate_path),
        resign_gate_fn=(lambda: os.path.exists(resign_gate_path)) if resign_gate_path else None,
        batch_games=batch_games,
        master_rng=master_rng,
        on_record=on_record,
        should_stop=stop_event.is_set,
        on_round=reload_ckpt,
        cfg_fn=cfg_fn if progress_path is not None else None,
    )
