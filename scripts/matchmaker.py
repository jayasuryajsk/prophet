"""Prophet's matchmaker: whenever idle, challenge a random online bot.
Runs alongside lichess_bot.py (which accepts and plays the games)."""

import json
import os
import random
import sys
import time

import requests

API = "https://lichess.org"
TOKEN = os.environ.get("LICHESS_TOKEN") or sys.exit("set LICHESS_TOKEN")
H = {"Authorization": f"Bearer {TOKEN}"}
ME = requests.get(f"{API}/api/account", headers=H, timeout=15).json()["id"]

CLOCKS = [(300, 3), (180, 2), (600, 5)]  # 5+3 blitz, 3+2 blitz, 10+5 rapid


def playing_now():
    r = requests.get(f"{API}/api/account/playing", headers=H, timeout=15).json()
    return len(r.get("nowPlaying", []))


def online_bots(limit=60):
    bots = []
    try:
        r = requests.get(f"{API}/api/bot/online", headers=H, stream=True, timeout=20)
        for line in r.iter_lines():
            if not line:
                continue
            b = json.loads(line)
            bots.append(b)
            if len(bots) >= limit:
                break
    except Exception:
        pass
    return bots


def pick(bots):
    cands = []
    for b in bots:
        if b.get("id") == ME:
            continue
        perfs = b.get("perfs", {})
        r = (perfs.get("blitz") or perfs.get("rapid") or {}).get("rating", 1500)
        if 1000 <= r <= 2600:
            cands.append((b["id"], r))
    return random.choice(cands) if cands else None


def main():
    print(f"matchmaker up as {ME}", flush=True)
    while True:
        try:
            if playing_now() == 0:
                got = pick(online_bots())
                if got:
                    op, r = got
                    limit, inc = random.choice(CLOCKS)
                    resp = requests.post(
                        f"{API}/api/challenge/{op}", headers=H,
                        data={"rated": "true", "clock.limit": limit, "clock.increment": inc},
                        timeout=20,
                    )
                    print(f"challenge {op} ({r}) {limit//60}+{inc}: {resp.status_code}"
                          + ("" if resp.status_code == 200 else " " + resp.text[:80]),
                          flush=True)
        except Exception as e:
            print(f"matchmaker error: {e}", flush=True)
        time.sleep(random.randint(45, 90))


if __name__ == "__main__":
    main()
