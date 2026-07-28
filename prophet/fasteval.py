"""Phase-A inference speedups for serving: transposition cache + int8 quant
+ optional torch.compile. Wraps the model transparently for any search.

The cache persists across moves within a game, so re-searched positions
(the bulk of consecutive searches) skip the net entirely — a poor-man's
tree reuse that composes with any search implementation.
"""

import hashlib

import torch
import torch.nn as nn


class FastEval(nn.Module):
    def __init__(self, model, cache_entries=30_000, quantize=False, compile_=False):
        super().__init__()
        if quantize:
            try:
                model = torch.ao.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                pass
        if compile_:
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                pass
        self.m = model
        self.cap = cache_entries
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def _key(self, x):
        return hashlib.blake2b(x.numpy().tobytes(), digest_size=16).digest()

    @torch.no_grad()
    def forward_wdl(self, x):
        k = self._key(x)
        hit = self.cache.get(k)
        if hit is not None:
            self.hits += 1
            return tuple(t.float() for t in hit)
        self.misses += 1
        out = self.m.forward_wdl(x)
        if len(self.cache) < self.cap:
            self.cache[k] = tuple(t.half() for t in out)
        return out

    def forward(self, x):
        p, a, v, _, _ = self.forward_wdl(x)
        return p, a, v

    def stats(self):
        n = self.hits + self.misses
        return f"cache {self.hits}/{n} hits ({100*self.hits/max(1,n):.0f}%)"
