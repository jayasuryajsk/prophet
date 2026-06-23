"""Baseline: does PURE-RANDOM-move pgx chess produce decisive games in 224 plies?

No network, no search, no prophet bridge -- just uniform-random legal moves in
raw pgx. This isolates whether the 0%-decisive is the ENV/cap (random also draws)
or the SEARCH/bridge (random decides games but prophet's search doesn't).
"""
import numpy as np
import jax
import jax.numpy as jnp
import pgx

env = pgx.make("chess")
B = 64
MAX_PLIES = 224
key = jax.random.PRNGKey(1)
key, ik = jax.random.split(key)
state = jax.vmap(env.init)(jax.random.split(ik, B))
step = jax.jit(jax.vmap(env.step))

done = jnp.zeros(B, dtype=bool)
end_ply = jnp.full(B, -1, dtype=jnp.int32)
decisive = jnp.zeros(B, dtype=bool)   # captured at the terminating step
for ply in range(MAX_PLIES):
    mask = state.legal_action_mask
    key, sub = jax.random.split(key)
    logits = jnp.where(mask, 0.0, -1e9)
    action = jax.random.categorical(sub, logits)
    state = step(state, action)
    newly = state.terminated & (~done)
    rmax = jnp.max(jnp.abs(state.rewards), axis=1)        # >0.5 => someone won
    decisive = jnp.where(newly, rmax > 0.5, decisive)
    end_ply = jnp.where(newly, ply + 1, end_ply)
    done = done | state.terminated

term = np.asarray(done)
dec = np.asarray(decisive)
ep = np.asarray(end_ply)
print(f"=== PURE-RANDOM pgx baseline (B={B}, max {MAX_PLIES} plies) ===", flush=True)
print(f"terminated within {MAX_PLIES} plies: {int(term.sum())}/{B}")
print(f"DECISIVE (someone won):             {int(dec.sum())}/{B}")
print(f"draws (terminated, nobody won):     {int((term & ~dec).sum())}/{B}")
print(f"still running at cap (no terminal): {int((~term).sum())}/{B}")
if term.any():
    print(f"end_ply of terminated games: min {ep[term].min()} mean {ep[term].mean():.1f} max {ep[term].max()}")
print("\nINTERPRETATION:")
print("  random pgx is DECISIVE >0  -> pgx CAN end games decisively; prophet's 0% = SEARCH/value bug")
print("  random pgx is ~0% decisive -> pgx chess just draws at 224 plies; need a lower cap / adjudication")
