"""Generation-pod entrypoint: play+reflect locally, stream lessons home.

Runs the UNCHANGED fast worker processes against a locally-mirrored copy
of the learner's control files (checkpoint / progress / gates — kept fresh
by ControlMirror over HTTP), and ships finished GameRecords to the learner
over an authenticated socket (RecordSink). Kill this pod any time: the
learner loses nothing but this node's in-flight games.

usage (recipe args must match the learner run):
  PROPHET_STREAM_KEY=... python3 scripts/gen_node.py \
      --control http://LEARNER:8000 --sink LEARNER:7000 \
      --workers 2 --threads 40
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: I001,F401  (torch before numpy; see README)

import argparse
import multiprocessing as mp
import os
import time

from prophet.stream import ControlMirror, RecordSink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="http://learner:port (run-dir file server)")
    ap.add_argument("--sink", required=True, help="learner host:port for game records")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--mega-batch", type=int, default=1024)
    ap.add_argument("--search-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-dir", default="/workspace/genmirror")
    ap.add_argument("--ckpt-every", type=float, default=20.0,
                    help="seconds between weight refreshes from the learner. "
                    "At swarm velocity, stale weights mean off-policy games — "
                    "90s cost ~10x learning efficiency at 94 g/min")
    # recipe args — MUST match the learner's run
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--contempt", type=float, default=0.15)
    ap.add_argument("--search-contempt", type=float, default=0.15)
    ap.add_argument("--win-discount", type=float, default=0.997)
    ap.add_argument("--pcr-prob", type=float, default=0.25)
    ap.add_argument("--pcr-cheap-sims", type=int, default=12)
    ap.add_argument("--study-topk", type=int, default=2)
    ap.add_argument("--deep-sims", type=int, default=128)
    ap.add_argument("--branch-plies", type=int, default=16)
    ap.add_argument("--study-weight", type=float, default=2.0)
    ap.add_argument("--no-study", action="store_true")
    ap.add_argument("--d-model", type=int, default=320)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    args = ap.parse_args()

    from prophet.fastplay import fast_vector_worker

    mirror = ControlMirror(args.control, args.local_dir,
                           ckpt_every=args.ckpt_every, poll=10.0)
    print(f"[gen] fetching initial weights from {args.control} ...", flush=True)
    mirror.start()
    print("[gen] control mirror live", flush=True)

    host, port = args.sink.rsplit(":", 1)
    sink = RecordSink(host, int(port))

    search_kwargs = {"sims": args.sims, "root_candidates": args.candidates,
                     "contempt": args.search_contempt}
    selfplay_kwargs = {
        "max_plies": args.max_plies, "contempt": args.contempt,
        "win_discount": args.win_discount, "pcr_prob": args.pcr_prob,
        "pcr_cheap_sims": args.pcr_cheap_sims,
    }
    study_kwargs = None if args.no_study else {
        "top_k": args.study_topk, "deep_sims": args.deep_sims,
        "branch_plies": args.branch_plies, "study_weight": args.study_weight,
    }
    model_kwargs = {"d_model": args.d_model, "n_layers": args.n_layers,
                    "n_heads": args.n_heads, "d_ff": 4 * args.d_model,
                    "head_dim": 64, "dropout": 0.0, "in_features": 24}

    d = args.local_dir
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue(maxsize=256)
    stop = ctx.Event()
    procs = [
        ctx.Process(
            target=fast_vector_worker,
            args=(i, os.path.join(d, "latest.pt"), os.path.join(d, "gate_on"),
                  out_q, stop, search_kwargs, selfplay_kwargs, study_kwargs,
                  model_kwargs, args.threads, args.mega_batch, args.device,
                  os.path.join(d, "progress.json")),
            kwargs={"resign_gate_path": os.path.join(d, "resign_on"),
                    "search_batch": args.search_batch,
                    "stats_path": os.path.join(d, "broker_stats.log")},
            daemon=True,
        )
        for i in range(args.workers)
    ]
    for p in procs:
        p.start()
    print(f"[gen] {args.workers} workers x {args.threads} threads on {args.device}", flush=True)

    n = 0
    t0 = time.time()
    try:
        while True:
            rec = out_q.get()
            sink.put(rec)
            n += 1
            if n % 25 == 0:
                el = (time.time() - t0) / 60
                print(f"[gen] {n} games | {n / max(el, 1e-9):.1f} g/min | "
                      f"shipped {sink.sent} dropped {sink.dropped}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        sink.stop()
        mirror.stop()


if __name__ == "__main__":
    main()
