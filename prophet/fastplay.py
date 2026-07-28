"""Rust-max self-play: threaded games on Rust search trees, one GPU broker.

The classic worker (worker.py) multiplexes generator searches one leaf per
game per round. This module is the throughput rewrite:

- Each game THREAD plays complete games with prophet_core.BatchSearch (the
  Phase C serving tree grown training-stats exports): 32 leaves per
  collect, virtual loss, sequential halving — tree ops in Rust with the
  GIL released, so threads genuinely run in parallel.
- One EvalBroker owns the model/device and coalesces every thread's leaf
  batches into GPU mega-batches (hundreds to a thousand positions per
  forward) — the regime where an accelerator actually pays.
- Sample/GameRecord construction is copied verbatim from selfplay.py /
  study.py (TD targets, PCR, truncation-as-draw, resignation, conversion
  study). The learner is untouched.

fast_vector_worker() is the process entrypoint, a drop-in sibling of
worker.vector_worker (same file-based gates/ckpt/progress protocol).
"""

import torch  # noqa: I001  (torch before numpy; see README)

import os
import queue as queue_mod
import threading
from dataclasses import replace

import numpy as np

import prophet_core

from .accel import autocast, setup_perf
from .encoding import FEATURES, NUM_ACTIONS
from .fastboard import board_from_fen, new_board
from .model import ModelConfig, PolicyQValueNet, extract_state
from .schedule import q_trust_at, study_config_at
from .search import SearchConfig, SearchResult
from .selfplay import GameRecord, Sample, SelfPlayConfig
from .study import StudyConfig, _sample_from_search, _top_line_indices, find_surprises

_NA = NUM_ACTIONS


# ---------------------------------------------------------------------------
# Eval broker: many threads submit leaf batches, one forward serves them all.
# ---------------------------------------------------------------------------


class EvalBroker:
    """Owns the model. Threads call eval(x[n,64,F]) -> packed [n, 2*4096+1]
    float32 (logits | adv | v). One pump thread drains the request queue,
    concatenates into a mega-batch, runs ONE forward, and scatters results.

    Weight reloads (and any schedule polling) run inside the pump thread
    between forwards via reload_fn — no locking needed anywhere."""

    def __init__(self, model, device, max_batch=1024, linger_s=0.0005,
                 reload_fn=None, reload_every=256):
        self.model = model
        self.device = device
        self.max_batch = max_batch
        self.linger_s = linger_s
        self.reload_fn = reload_fn
        self.reload_every = reload_every
        self.q = queue_mod.Queue()
        self.stopped = threading.Event()
        self.error = None  # set if the pump dies; eval() raises instead of hanging
        self.batches = 0
        self.positions = 0
        self._thread = threading.Thread(target=self._pump_guard, daemon=True)

    def start(self):
        if self.reload_fn is not None:
            self.reload_fn()
        self._thread.start()

    def stop(self):
        self.stopped.set()

    def eval(self, x: np.ndarray) -> np.ndarray:
        req = [x, threading.Event(), None]
        self.q.put(req)
        req[1].wait()
        if req[2] is None:
            raise RuntimeError(f"eval broker died: {self.error!r}")
        return req[2]

    def _pump_guard(self):
        # a dead pump must CRASH the waiters, not strand them: on any pump
        # exception (GPU fault, OOM, ...) release every queued request with a
        # None result so eval() raises loudly in each game thread.
        try:
            self._pump()
        except BaseException as e:  # noqa: BLE001
            self.error = e
            self.stopped.set()
        while True:
            try:
                r = self.q.get_nowait()
            except queue_mod.Empty:
                break
            r[1].set()

    def _drain(self, reqs, rows):
        while rows < self.max_batch:
            try:
                r = self.q.get_nowait()
            except queue_mod.Empty:
                break
            reqs.append(r)
            rows += len(r[0])
        return rows

    def _pump(self):
        model, device = self.model, self.device
        while not self.stopped.is_set():
            try:
                first = self.q.get(timeout=0.2)
            except queue_mod.Empty:
                continue
            reqs = [first]
            rows = self._drain(reqs, len(first[0]))
            if rows < self.max_batch // 4 and self.linger_s > 0:
                # nearly-empty round: wait a hair for stragglers so the GPU
                # sees fuller batches (a no-op under real load — the queue
                # is never this empty when all threads are searching)
                self.stopped.wait(self.linger_s)
                rows = self._drain(reqs, rows)
            x = reqs[0][0] if len(reqs) == 1 else np.concatenate([r[0] for r in reqs])
            xt = torch.from_numpy(np.ascontiguousarray(x))
            with torch.inference_mode(), autocast(device):
                logits, adv, v = model(xt.to(device, non_blocking=True))
            # one packed D2H transfer (see worker.py): [B,4096]|[B,4096]|[B,1]
            arr = torch.cat([logits, adv, v[:, None]], dim=1).float().cpu().numpy()
            off = 0
            for r in reqs:
                n = len(r[0])
                r[2] = arr[off:off + n]
                off += n
                r[1].set()
            self.batches += 1
            self.positions += rows
            if self.reload_fn is not None and self.batches % self.reload_every == 0:
                try:
                    self.reload_fn()
                except Exception:
                    pass  # mid-write ckpt etc.; retry next interval


# ---------------------------------------------------------------------------
# One search on the Rust tree -> a full training SearchResult.
# ---------------------------------------------------------------------------


def _packed_to_bytes(arr):
    lg = np.ascontiguousarray(arr[:, :_NA]).tobytes()
    ad = np.ascontiguousarray(arr[:, _NA:2 * _NA]).tobytes()
    vs = np.ascontiguousarray(arr[:, 2 * _NA]).tobytes()
    return lg, ad, vs


def rust_search(fen: str, cfg: SearchConfig, rng, eval_fn, batch=32) -> SearchResult:
    """Run one Gumbel search in Rust; eval_fn(x[n,64,F]) -> packed [n,8193].
    budget = sims+1 so post-root playouts exactly match the python search's
    cfg.sims (the Rust tree counts the root eval as spent)."""
    seed = int(rng.integers(1, 1 << 62))
    t = prophet_core.BatchSearch(
        fen, cfg.sims + 1, min(batch, max(1, cfg.sims)), cfg.root_candidates,
        cfg.c_puct, cfg.c_visit, cfg.c_scale, cfg.q_trust, cfg.contempt, seed,
    )
    x = np.asarray(t.root_features(), dtype=np.float32).reshape(1, 64, FEATURES)
    lg, ad, vs = _packed_to_bytes(eval_fn(x))
    t.set_root(lg, ad, np.frombuffer(vs, dtype=np.float32)[0].item())
    guard = 0
    while not t.done():
        fb = t.collect()
        n = t.n_pending()
        if n == 0:
            guard += 1
            if guard > 4 * cfg.sims + 8:
                break
            continue
        guard = 0
        x = np.frombuffer(fb, dtype=np.float32).reshape(n, 64, FEATURES).copy()
        t.apply(*_packed_to_bytes(eval_fn(x)))
    best = int(t.best())
    acts, probs = t.policy_target()
    qa, qq, qn = t.visited_children()
    return SearchResult(
        move_index=best,
        root_value=float(t.root_value()),
        legal_indices=np.asarray(acts, dtype=np.int64),
        policy_target=np.asarray(probs, dtype=np.float32),
        q_indices=np.asarray(qa, dtype=np.int64),
        q_values=np.asarray(qq, dtype=np.float32),
        q_visits=np.asarray(qn, dtype=np.float32),
        q_head_played=float(t.root_q_raw(best)),
    )


# ---------------------------------------------------------------------------
# Self-play + study, plain functions (ported 1:1 from the generator forms in
# selfplay.play_game_gen / study.study_game_gen — same math, Rust searches).
# ---------------------------------------------------------------------------


def play_game_fast(
    search_cfg: SearchConfig,
    cfg: SelfPlayConfig,
    rng: np.random.Generator,
    eval_fn,
    resign_enabled: bool = False,
    batch: int = 32,
) -> GameRecord:
    board = new_board()
    raw = []  # (x, search result, child features, mover_was_white, full_search)
    fens = []
    resign_active = resign_enabled and rng.random() >= cfg.resign_off_prob
    low_streak = {True: 0, False: 0}
    resigned_winner_white = None
    cheap_cfg = (
        replace(
            search_cfg,
            sims=cfg.pcr_cheap_sims,
            root_candidates=min(search_cfg.root_candidates, 4),
        )
        if cfg.pcr_prob > 0
        else None
    )

    while board.terminal_value() is None and len(raw) < cfg.max_plies:
        mover_white = board.turn
        fens.append(board.fen())
        x = board.encode()
        full = cheap_cfg is None or rng.random() < cfg.pcr_prob
        res = rust_search(fens[-1], search_cfg if full else cheap_cfg, rng, eval_fn, batch)
        board.push_action(res.move_index)
        child_x = board.encode()
        raw.append((x, res, child_x, mover_white, full))

        if res.root_value < cfg.resign_threshold:
            low_streak[mover_white] += 1
        else:
            low_streak[mover_white] = 0
        if resign_active and low_streak[mover_white] >= cfg.resign_plies:
            resigned_winner_white = not mover_white
            break

    end_known = True
    if resigned_winner_white is not None:
        result = "1-0" if resigned_winner_white else "0-1"
        z_white = 1.0 if resigned_winner_white else -1.0
    else:
        term = board.terminal_value()
        if term is None:
            result = "*"
            z_white = 0.0
            end_known = False
        elif term == -1.0:
            result = "0-1" if board.turn else "1-0"
            z_white = -1.0 if board.turn else 1.0
        else:
            result = "1/2-1/2"
            z_white = 0.0

    samples = []
    total = len(raw)
    for t, (x, res, child_x, mover_was_white, full_search) in enumerate(raw):
        v = res.root_value
        wdl = -1
        if z_white is not None:
            z = z_white if mover_was_white else -z_white
            wdl = int(z) + 1
            if z == 0.0:
                z_eff = -cfg.contempt
            else:
                z_eff = z * cfg.win_discount ** (total - t)
            n = cfg.td_steps
            if n > 0 and t + n < total:
                v_ahead = raw[t + n][1].root_value
                boot = (v_ahead if n % 2 == 0 else -v_ahead) * cfg.win_discount**n
                z_eff = (1 - cfg.td_outcome_leak) * boot + cfg.td_outcome_leak * z_eff
            v = (1 - cfg.outcome_mix) * v + cfg.outcome_mix * z_eff
        samples.append(
            Sample(
                x=x,
                legal_indices=res.legal_indices,
                policy_target=res.policy_target,
                value_target=float(v),
                q_indices=res.q_indices,
                q_values=res.q_values,
                q_visits=res.q_visits,
                played_index=res.move_index,
                child_x=child_x,
                wdl=wdl,
                moves_left=float(total - t) if end_known else -1.0,
                policy_ok=full_search,
            )
        )
    return GameRecord(
        samples=samples,
        result=result,
        plies=len(raw),
        fens=fens,
        root_values=[res.root_value for _, res, *_ in raw],
        q_head_played=[res.q_head_played for _, res, *_ in raw],
    )


def _play_branch_fast(board, scfg, cfg: StudyConfig, rng, eval_fn, batch):
    raw = []
    while board.terminal_value() is None and len(raw) < cfg.branch_plies:
        res = rust_search(board.fen(), scfg, rng, eval_fn, batch)
        raw.append(
            (
                _sample_from_search(board, res, res.root_value, cfg.branch_weight),
                board.turn,
            )
        )
        board.push_action(res.move_index)
    term = board.terminal_value()
    if term is not None:
        if term == -1.0:
            z_white = -1.0 if board.turn else 1.0
        else:
            z_white = 0.0
        for s, mover_was_white in raw:
            z = z_white if mover_was_white else -z_white
            if z == 0.0:
                z = -cfg.contempt
            s.value_target = (1 - cfg.outcome_mix) * s.value_target + cfg.outcome_mix * z
    return [s for s, _ in raw]


def study_game_fast(
    record: GameRecord,
    scfg: SearchConfig,
    cfg: StudyConfig,
    rng: np.random.Generator,
    eval_fn,
    batch: int = 32,
) -> list:
    if not record.fens:
        return []
    deep_cfg = SearchConfig(
        sims=cfg.deep_sims,
        root_candidates=cfg.deep_candidates,
        q_trust=scfg.q_trust,
        contempt=scfg.contempt,
    )
    out = []
    telem = os.environ.get("PROPHET_STUDY_LOG")
    for t, kind in find_surprises(record, cfg):
        board = board_from_fen(record.fens[t])
        res = rust_search(record.fens[t], deep_cfg, rng, eval_fn, batch)
        out.append(_sample_from_search(board, res, res.root_value, cfg.study_weight))
        bcfg = replace(cfg, branch_plies=cfg.conv_branch_plies) if kind == "conv" else cfg
        n_br = n_term = 0
        for mv_idx in _top_line_indices(res, cfg.n_lines):
            branch_board = board_from_fen(record.fens[t])
            branch_board.push_action(int(mv_idx))
            out.extend(_play_branch_fast(branch_board, scfg, bcfg, rng, eval_fn, batch))
            n_br += 1
            n_term += branch_board.terminal_value() is not None
        if telem:
            try:
                with open(telem, "a") as f:
                    f.write(
                        f"{kind} ply={t}/{record.plies} res={record.result} term={n_term}/{n_br}\n"
                    )
            except OSError:
                pass
    return out


# ---------------------------------------------------------------------------
# Process entrypoint: broker + N game threads. Same file protocol as
# worker.vector_worker (ckpt mtime sync, gate files, progress schedule).
# ---------------------------------------------------------------------------


def fast_vector_worker(
    worker_id: int,
    ckpt_path: str,
    gate_path: str,
    out_queue,
    stop_event,
    search_kwargs: dict,
    selfplay_kwargs: dict,
    study_kwargs: dict | None,
    model_kwargs: dict,
    n_threads: int = 16,
    mega_batch: int = 512,
    device_str: str = "cpu",
    progress_path: str | None = None,
    search_batch: int = 32,
    resign_gate_path: str | None = None,
    stats_path: str | None = None,
):
    torch.set_num_threads(max(2, os.cpu_count() // 4) if device_str == "cpu" else 2)
    setup_perf(device_str)
    device = torch.device(device_str)
    model = PolicyQValueNet(ModelConfig(**model_kwargs)).to(device)
    model.eval()
    scfg = SearchConfig(**search_kwargs)
    spcfg = SelfPlayConfig(**selfplay_kwargs)
    stcfg = StudyConfig(**study_kwargs) if study_kwargs is not None else None

    state = {"mtime": 0.0, "games": 0}

    def reload_ckpt():
        try:
            m = os.path.getmtime(ckpt_path)
            if m != state["mtime"]:
                model.load_state_dict(extract_state(ckpt_path))
                state["mtime"] = m
        except (OSError, RuntimeError, KeyError):
            pass  # mid-write or missing; retry next interval
        if progress_path is not None:
            try:
                with open(progress_path) as f:
                    import json

                    state["games"] = int(json.load(f).get("games", 0))
            except (OSError, ValueError):
                pass

    broker = EvalBroker(model, device, max_batch=mega_batch, reload_fn=reload_ckpt)
    broker.start()

    def cfg_fn():
        if progress_path is None:
            return scfg, stcfg
        games = state["games"]
        s_cfg = replace(scfg, q_trust=q_trust_at(games))
        st_cfg = study_config_at(games, stcfg) if stcfg is not None else None
        return s_cfg, st_cfg

    gate_fn = lambda: os.path.exists(gate_path)  # noqa: E731
    resign_fn = (
        (lambda: os.path.exists(resign_gate_path)) if resign_gate_path else gate_fn
    )

    def offer(record) -> bool:
        while not stop_event.is_set():
            try:
                out_queue.put(record, timeout=1.0)
                return True
            except queue_mod.Full:
                continue
        return False

    def game_thread(tid: int):
        rng = np.random.default_rng([worker_id, tid, os.getpid()])
        while not stop_event.is_set():
            try:
                s_cfg, st_cfg = cfg_fn()
                record = play_game_fast(
                    s_cfg, spcfg, rng, broker.eval,
                    resign_enabled=resign_fn(), batch=search_batch,
                )
                if st_cfg is not None and gate_fn():
                    record.samples.extend(
                        study_game_fast(record, s_cfg, st_cfg, rng, broker.eval, batch=search_batch)
                    )
            except Exception as e:  # noqa: BLE001
                if broker.error is not None or stop_event.is_set():
                    return  # broker down / shutting down: exit quietly
                print(f"[fastworker {worker_id}.{tid}] game error, skipping: {e!r}", flush=True)
                continue
            if not offer(record):
                return

    threads = [
        threading.Thread(target=game_thread, args=(tid,), daemon=True)
        for tid in range(n_threads)
    ]
    for t in threads:
        t.start()
    # main thread: wait for stop, occasionally report broker load
    import time as _time

    last = (0, 0, _time.time())
    while not stop_event.is_set():
        stop_event.wait(30.0)
        if stats_path is not None:
            b, p, t0 = last
            now = _time.time()
            db, dp = broker.batches - b, broker.positions - p
            try:
                with open(stats_path, "a") as f:
                    f.write(
                        f"{now:.0f} worker{worker_id} evals/s={dp / max(1e-9, now - t0):.0f} "
                        f"fill={dp / max(1, db):.0f}/{mega_batch}\n"
                    )
            except OSError:
                pass
            last = (broker.batches, broker.positions, now)
    broker.stop()
