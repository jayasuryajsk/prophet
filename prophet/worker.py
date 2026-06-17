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

from .model import ModelConfig, PolicyQValueNet, extract_state
from .schedule import q_trust_at, study_config_at
from .search import SearchConfig
from .selfplay import SelfPlayConfig, play_game_gen
from .study import StudyConfig, study_game_gen

RELOAD_EVERY_ROUNDS = 32


def episode_gen(scfg, spcfg, stcfg, rng, gate_fn):
    """One full episode: a game, plus study if the gate is open."""
    gate = gate_fn()
    record = yield from play_game_gen(scfg, spcfg, rng, resign_enabled=gate)
    if stcfg is not None and gate:
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
):
    """Core driver loop. Calls on_record(record) for each finished episode;
    runs until should_stop() is true. cfg_fn(), if given, returns the
    (scfg, stcfg) to use for each new episode (game-count curricula)."""

    def new_episode():
        rng = np.random.default_rng(int(master_rng.integers(2**63)))
        s_cfg, st_cfg = cfg_fn() if cfg_fn is not None else (scfg, stcfg)
        gen = episode_gen(s_cfg, spcfg, st_cfg, rng, gate_fn)
        return gen, gen.send(None)

    gens, pending = [], []
    for _ in range(batch_games):
        g, x = new_episode()
        gens.append(g)
        pending.append(x)

    rounds = 0
    while not should_stop():
        if on_round is not None and rounds % RELOAD_EVERY_ROUNDS == 0:
            on_round(rounds)
        xb = torch.from_numpy(np.stack(pending)).to(device)
        with torch.no_grad():
            logits, q, v = model(xb)
        logits = logits.cpu().numpy()
        q = q.cpu().numpy()
        v = v.cpu().numpy()
        for i in range(len(gens)):
            try:
                pending[i] = gens[i].send((logits[i], q[i], v[i]))
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
):
    torch.set_num_threads(threads)
    device = torch.device(device_str)
    model = PolicyQValueNet(ModelConfig(**model_kwargs)).to(device)
    model.eval()
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
        batch_games=batch_games,
        master_rng=master_rng,
        on_record=on_record,
        should_stop=stop_event.is_set,
        on_round=reload_ckpt,
        cfg_fn=cfg_fn if progress_path is not None else None,
    )
