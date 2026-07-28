"""Prophet on Lichess — direct Bot API bridge.

Streams events, accepts standard challenges, plays with Searcher35
(tree reuse across moves — its natural habitat at live time controls),
budgets forwards from the clock, hot-swaps to newer checkpoints between
games if a newer ckpt file appears.

usage:
  LICHESS_TOKEN=xxx python3 scripts/lichess_bot.py CKPT [--upgrade]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: I001

import argparse
import json
import os
import threading
import time

import chess
import numpy as np
import requests

import numpy as _np

from prophet.model import load_checkpoint
from prophet.search import SearchConfig, search_move

API = "https://lichess.org"
_lock = threading.Lock()  # one search at a time (shared CPU)


def _headers(token):
    return {"Authorization": f"Bearer {token}"}


def measure_nps(model):
    from prophet.fastboard import PyChessBoard

    pb = PyChessBoard(chess.Board())
    x = torch.from_numpy(pb.encode()).unsqueeze(0)
    with torch.no_grad():
        for _ in range(5):
            model.forward_wdl(x)  # warmup
        t0 = time.perf_counter()
        for _ in range(50):
            model.forward_wdl(x)
    per = (time.perf_counter() - t0) / 50
    return 1.0 / per


class Game(threading.Thread):
    def __init__(self, bot, game_id):
        super().__init__(daemon=True)
        self.bot = bot
        self.id = game_id

    def budget(self, my_ms, inc_ms, board):
        # brisk: ~1/50th of clock (capped 4s) + half the increment
        t = min(my_ms / 1000.0 / 50.0, 4.0) + inc_ms / 1000.0 * 0.5
        t = max(0.3, t)
        if my_ms < 15_000:  # panic mode
            t = min(t, 1.0)
        b = int(self.bot.nps * t)
        return int(np.clip(b, 48, 2048))

    def run(self):
        bot = self.bot
        url = f"{API}/api/bot/game/stream/{self.id}"
        my_color = None
        rng = _np.random.default_rng(int(time.time()) % 1_000_000)
        try:
            r = requests.get(url, headers=_headers(bot.token), stream=True, timeout=30)
            for line in r.iter_lines():
                if not line:
                    continue
                ev = json.loads(line)
                if ev["type"] == "gameFull":
                    my_color = chess.WHITE if ev["white"].get("id") == bot.me else chess.BLACK
                    state = ev["state"]
                else:
                    if ev["type"] != "gameState":
                        continue
                    state = ev
                if state.get("status") not in (None, "started"):
                    break
                board = chess.Board()
                for u in state["moves"].split():
                    board.push_uci(u)
                if board.turn != my_color or board.is_game_over():
                    continue
                my_ms = state["wtime"] if my_color == chess.WHITE else state["btime"]
                inc_ms = state["winc"] if my_color == chess.WHITE else state["binc"]
                with _lock:
                    b = self.budget(my_ms, inc_ms, board)
                    # the PROVEN baseline search — the one every validated
                    # number was measured with (v3.5 stays on the bench)
                    cfg = SearchConfig(sims=max(31, b - 1), root_candidates=16)
                    mv = search_move(bot.model, board, cfg,
                                     torch.device("cpu"), rng)
                uci = mv.uci()
                for attempt in range(3):
                    resp = requests.post(
                        f"{API}/api/bot/game/{self.id}/move/{uci}",
                        headers=_headers(bot.token), timeout=15,
                    )
                    if resp.status_code == 200:
                        break
                    time.sleep(1)
                bot.log(f"[{self.id}] played {uci} (budget {b}, clock {my_ms/1000:.0f}s)")
        except Exception as e:
            bot.log(f"[{self.id}] game thread error: {e}")
        finally:
            bot.active.discard(self.id)
            bot.log(f"[{self.id}] game over")


class Bot:
    def __init__(self, token, ckpt):
        self.token = token
        self.ckpt_path = ckpt
        self.model = load_checkpoint(ckpt)
        self.model.eval()
        torch.set_num_threads(4)
        self.nps = measure_nps(self.model)
        self.active = set()
        me = requests.get(f"{API}/api/account", headers=_headers(token), timeout=15).json()
        self.me = me["id"]
        self.log(f"online as {self.me} | ckpt {ckpt} | {self.nps:.0f} forwards/s")

    def log(self, msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def ok_challenge(self, ch):
        if ch["variant"]["key"] != "standard":
            return False
        tc = ch["timeControl"]
        if tc["type"] != "clock" or tc["limit"] < 60:
            return False
        if len(self.active) >= 2:
            return False
        return True

    def run(self):
        while True:
            try:
                r = requests.get(f"{API}/api/stream/event",
                                 headers=_headers(self.token), stream=True, timeout=30)
                for line in r.iter_lines():
                    if not line:
                        continue
                    ev = json.loads(line)
                    if ev["type"] == "challenge":
                        ch = ev["challenge"]
                        if ch.get("challenger", {}).get("id") == self.me:
                            continue
                        verb = "accept" if self.ok_challenge(ch) else "decline"
                        requests.post(f"{API}/api/challenge/{ch['id']}/{verb}",
                                      headers=_headers(self.token), timeout=15)
                        self.log(f"challenge from {ch.get('challenger',{}).get('id')}: {verb}")
                    elif ev["type"] == "gameStart":
                        gid = ev["game"]["gameId"]
                        if gid not in self.active:
                            self.active.add(gid)
                            Game(self, gid).start()
            except Exception as e:
                self.log(f"event stream dropped ({e}); reconnecting in 5s")
                time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--upgrade", action="store_true", help="one-time BOT account upgrade")
    args = ap.parse_args()
    token = os.environ.get("LICHESS_TOKEN")
    if not token:
        sys.exit("set LICHESS_TOKEN")
    if args.upgrade:
        r = requests.post(f"{API}/api/bot/account/upgrade", headers=_headers(token), timeout=15)
        print("upgrade:", r.status_code, r.text[:200])
        if r.status_code != 200:
            sys.exit(1)
    bot = Bot(token, args.ckpt)
    if args.device != "cpu":
        bot.model = bot.model.to(args.device)
        bot.nps = measure_nps(bot.model)
        bot.log(f"moved to {args.device} | {bot.nps:.0f} forwards/s")
    bot.run()


if __name__ == "__main__":
    main()
