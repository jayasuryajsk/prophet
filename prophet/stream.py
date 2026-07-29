"""Fleet transport: stream GameRecords from generation pods to the learner.

Design goals, in order: (1) the validated worker/learner code runs verbatim
on both ends — this module is pure plumbing; (2) generation pods are
disposable — a dead one costs its in-flight games, nothing else; (3) the
learner never blocks on the network.

Learner side: record_server() accepts authenticated connections and feeds
received GameRecords into the SAME queue local workers use.

Gen side: RecordSink.put(record) ships a record, buffering a little and
reconnecting forever — generation never dies because the learner blipped.

Transport: multiprocessing.connection (length-prefixed pickle + HMAC
challenge on connect) — the network twin of the mp.Queue the single-box
pipeline already trusts.
"""

import os
import queue as queue_mod
import threading
import time
from multiprocessing.connection import Client, Listener


def _authkey():
    k = os.environ.get("PROPHET_STREAM_KEY", "")
    if not k:
        raise RuntimeError("set PROPHET_STREAM_KEY (shared secret) on both ends")
    return k.encode()


# ---------------------------------------------------------------------------
# learner side
# ---------------------------------------------------------------------------


def record_server(port, game_q, stop_event, stats=None):
    """Accept gen-node connections; pump their records into game_q.
    Runs until stop_event. One thread per connection; a dead/misbehaving
    connection is dropped without ceremony."""
    listener = Listener(("0.0.0.0", port), authkey=_authkey())
    stats = stats if stats is not None else {}
    stats.setdefault("conns", 0)
    stats.setdefault("records", 0)

    def serve(conn, peer):
        stats["conns"] += 1
        try:
            while not stop_event.is_set():
                if not conn.poll(1.0):
                    continue
                rec = conn.recv()
                while not stop_event.is_set():
                    try:
                        game_q.put(rec, timeout=1.0)
                        stats["records"] += 1
                        break
                    except queue_mod.Full:
                        continue
        except (EOFError, OSError, ConnectionError):
            pass
        finally:
            stats["conns"] -= 1
            try:
                conn.close()
            except OSError:
                pass

    def accept_loop():
        while not stop_event.is_set():
            try:
                conn = listener.accept()
                peer = listener.last_accepted
                threading.Thread(target=serve, args=(conn, peer), daemon=True).start()
            except (OSError, EOFError, Exception):  # auth failures land here too
                if stop_event.is_set():
                    return
                time.sleep(0.5)

    t = threading.Thread(target=accept_loop, daemon=True)
    t.start()
    return stats


# ---------------------------------------------------------------------------
# gen side
# ---------------------------------------------------------------------------


class RecordSink:
    """Queue-shaped shipper: .put(record) never blocks generation for long.
    Buffers up to `buffer` records across outages, drops oldest beyond
    (generation pods are loss-tolerant by design)."""

    def __init__(self, host, port, buffer=512):
        self.addr = (host, port)
        self.q = queue_mod.Queue(maxsize=buffer)
        self.stopped = threading.Event()
        self.sent = 0
        self.dropped = 0
        self._t = threading.Thread(target=self._ship, daemon=True)
        self._t.start()

    def put(self, record, timeout=None):
        try:
            self.q.put_nowait(record)
        except queue_mod.Full:
            try:
                self.q.get_nowait()  # drop oldest
                self.dropped += 1
                self.q.put_nowait(record)
            except queue_mod.Empty:
                pass
        return True

    def _connect(self):
        while not self.stopped.is_set():
            try:
                return Client(self.addr, authkey=_authkey())
            except (ConnectionError, OSError):
                time.sleep(3.0)
        return None

    def _ship(self):
        conn = None
        pending = None
        while not self.stopped.is_set():
            if conn is None:
                conn = self._connect()
                if conn is None:
                    return
            try:
                if pending is None:
                    try:
                        pending = self.q.get(timeout=1.0)
                    except queue_mod.Empty:
                        continue
                conn.send(pending)
                self.sent += 1
                pending = None  # only clear after a successful send
            except (BrokenPipeError, ConnectionError, OSError):
                try:
                    conn.close()
                except OSError:
                    pass
                conn = None  # reconnect; `pending` re-sends on the new conn

    def stop(self):
        self.stopped.set()


class ControlMirror:
    """Recreates the learner's file environment on a gen pod.

    Polls the learner's HTTP server (a bare http.server in the run dir) and
    maintains local twins: latest.pt (atomic replace -> workers' mtime
    reload just works), progress.json, and gate touch-files. The worker
    processes cannot tell they are not on the learner box."""

    def __init__(self, base_url, local_dir, ckpt_every=90.0, poll=20.0):
        self.base = base_url.rstrip("/")
        self.dir = local_dir
        self.ckpt_every = ckpt_every
        self.poll = poll
        self.stopped = threading.Event()
        os.makedirs(local_dir, exist_ok=True)
        self._last_ckpt = 0.0
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._fetch_ckpt()  # block until we have initial weights
        self._fetch_control()
        self._t.start()

    def _get(self, name):
        import urllib.request

        with urllib.request.urlopen(f"{self.base}/{name}", timeout=60) as r:
            return r.read()

    def _fetch_ckpt(self):
        data = self._get("latest.pt")
        tmp = os.path.join(self.dir, "latest.pt.tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, os.path.join(self.dir, "latest.pt"))
        self._last_ckpt = time.time()

    def _fetch_control(self):
        import json

        try:
            ctl = json.loads(self._get("progress.json"))
        except Exception:
            return
        tmp = os.path.join(self.dir, "progress.json.tmp")
        with open(tmp, "w") as f:
            json.dump(ctl, f)
        os.replace(tmp, os.path.join(self.dir, "progress.json"))
        for key, fname in (("study_gate", "gate_on"), ("resign_gate", "resign_on")):
            path = os.path.join(self.dir, fname)
            if ctl.get(key):
                open(path, "a").close()
            # gates only ever open; no removal path needed

    def _loop(self):
        while not self.stopped.is_set():
            self.stopped.wait(self.poll)
            try:
                self._fetch_control()
                if time.time() - self._last_ckpt >= self.ckpt_every:
                    self._fetch_ckpt()
            except Exception:
                pass  # learner blip; keep generating on current weights

    def stop(self):
        self.stopped.set()
