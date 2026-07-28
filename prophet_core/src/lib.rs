// Fast chess core for prophet: move-gen, board ops, terminal detection, and
// the exact 24-feature side-to-move encoding (must match encode_board in
// prophet/encoding.py bit-for-bit — validated by scripts/validate_core.py).

use pyo3::prelude::*;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{
    CastlingMode, CastlingSide, Chess, Color, EnPassantMode, File, Move, Position, Rank, Role,
    Square,
};
use shakmaty::fen::Fen;

const FEATURES: usize = 24;
const DRAW_HALFMOVE_CAP: u32 = 100;

fn role_plane(role: Role) -> usize {
    match role {
        Role::Pawn => 0,
        Role::Knight => 1,
        Role::Bishop => 2,
        Role::Rook => 3,
        Role::Queen => 4,
        Role::King => 5,
    }
}

// Normalized (from, to) squares as 0..63, castling as king e1->g1/c1 (UCI
// standard), matching python-chess move_to_index before the perspective flip.
fn move_squares(m: &Move) -> (u32, u32) {
    match m {
        Move::Castle { king, rook } => {
            let side = if rook.file() > king.file() {
                CastlingSide::KingSide
            } else {
                CastlingSide::QueenSide
            };
            let to_file = if side == CastlingSide::KingSide {
                File::G
            } else {
                File::C
            };
            (u32::from(*king), u32::from(Square::from_coords(to_file, king.rank())))
        }
        _ => (u32::from(m.from().unwrap()), u32::from(m.to())),
    }
}

#[pyclass]
pub struct Board {
    stack: Vec<Chess>,
    moves: Vec<Move>,
    hashes: Vec<u64>,
}

impl Board {
    fn cur(&self) -> &Chess {
        self.stack.last().unwrap()
    }
    fn flip(&self) -> u32 {
        if self.cur().turn() == Color::Black {
            56
        } else {
            0
        }
    }
    fn hash_of(pos: &Chess) -> u64 {
        let z: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
        z.0
    }
    fn rep_count(&self) -> usize {
        let h = *self.hashes.last().unwrap();
        self.hashes.iter().filter(|&&x| x == h).count()
    }
}

#[pymethods]
impl Board {
    #[new]
    fn new() -> Self {
        let pos = Chess::default();
        let h = Board::hash_of(&pos);
        Board {
            stack: vec![pos],
            moves: vec![],
            hashes: vec![h],
        }
    }

    #[staticmethod]
    fn from_fen(fen: &str) -> PyResult<Self> {
        let parsed: Fen = fen
            .parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad fen: {e}")))?;
        let pos: Chess = parsed
            .into_position(CastlingMode::Standard)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad pos: {e}")))?;
        let h = Board::hash_of(&pos);
        Ok(Board {
            stack: vec![pos],
            moves: vec![],
            hashes: vec![h],
        })
    }

    fn fen(&self) -> String {
        Fen::from_position(self.cur().clone(), EnPassantMode::Always).to_string()
    }

    #[getter]
    fn turn(&self) -> bool {
        self.cur().turn() == Color::White
    }

    // Action indices (from*64 + to, side-to-move flipped), queen-promo only.
    fn legal_actions(&self) -> Vec<u16> {
        let flip = self.flip();
        let mut out = Vec::new();
        for m in self.cur().legal_moves() {
            if let Some(p) = m.promotion() {
                if p != Role::Queen {
                    continue;
                }
            }
            let (f, t) = move_squares(&m);
            out.push(((f ^ flip) * 64 + (t ^ flip)) as u16);
        }
        out
    }

    fn push_action(&mut self, action: u16) -> PyResult<()> {
        let flip = self.flip();
        let a = action as u32;
        let f = (a / 64) ^ flip;
        let t = (a % 64) ^ flip;
        let mut chosen: Option<Move> = None;
        for m in self.cur().legal_moves() {
            if let Some(p) = m.promotion() {
                if p != Role::Queen {
                    continue;
                }
            }
            let (mf, mt) = move_squares(&m);
            if mf == f && mt == t {
                chosen = Some(m);
                break;
            }
        }
        let mv = chosen.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("illegal action {action}"))
        })?;
        let next = self.cur().clone().play(&mv).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("play error: {e}"))
        })?;
        // (kept fallible for the Python-facing API; search paths use known-legal
        // actions and ignore the Result)
        let h = Board::hash_of(&next);
        self.stack.push(next);
        self.moves.push(mv);
        self.hashes.push(h);
        Ok(())
    }

    fn pop(&mut self) {
        if self.stack.len() > 1 {
            self.stack.pop();
            self.moves.pop();
            self.hashes.pop();
        }
    }

    // Value for the side to move, or None. Matches _terminal_value.
    fn terminal_value(&self) -> Option<f32> {
        let pos = self.cur();
        if pos.is_checkmate() {
            return Some(-1.0);
        }
        if pos.is_stalemate()
            || pos.is_insufficient_material()
            || pos.halfmoves() >= DRAW_HALFMOVE_CAP
            || self.rep_count() >= 3
        {
            return Some(0.0);
        }
        None
    }

    // 64*24 features, row-major [square, plane], side-to-move perspective.
    fn encode(&self) -> Vec<f32> {
        let pos = self.cur();
        let board = pos.board();
        let us = pos.turn();
        let flip = self.flip();
        let mut x = vec![0.0f32; 64 * FEATURES];

        for sq in Square::ALL {
            if let Some(role) = board.role_at(sq) {
                let color = board.color_at(sq).unwrap();
                let base = if color == us { 0 } else { 6 };
                let plane = base + role_plane(role);
                let idx = ((u32::from(sq) ^ flip) as usize) * FEATURES + plane;
                x[idx] = 1.0;
            }
        }
        if let Some(ep) = pos.ep_square(EnPassantMode::Always) {
            x[((u32::from(ep) ^ flip) as usize) * FEATURES + 12] = 1.0;
        }
        let them = us.other();
        let castles = pos.castles();
        let rights = [
            castles.rook(us, CastlingSide::KingSide).is_some(),
            castles.rook(us, CastlingSide::QueenSide).is_some(),
            castles.rook(them, CastlingSide::KingSide).is_some(),
            castles.rook(them, CastlingSide::QueenSide).is_some(),
        ];
        let hm = pos.halfmoves().min(100) as f32 / 100.0;
        let rep = if pos.halfmoves() >= 4 && self.rep_count() >= 2 {
            1.0
        } else {
            0.0
        };
        let parity = (self.moves.len() % 2) as f32;
        for sq in 0..64 {
            let row = sq * FEATURES;
            for k in 0..4 {
                if rights[k] {
                    x[row + 13 + k] = 1.0;
                }
            }
            x[row + 17] = hm;
            x[row + 22] = rep;
            x[row + 23] = parity;
        }
        // history: last two moves' from/to as flipped square flags
        let n = self.moves.len();
        if n >= 1 {
            let (f, t) = move_squares(&self.moves[n - 1]);
            x[((f ^ flip) as usize) * FEATURES + 18] = 1.0;
            x[((t ^ flip) as usize) * FEATURES + 19] = 1.0;
        }
        if n >= 2 {
            let (f, t) = move_squares(&self.moves[n - 2]);
            x[((f ^ flip) as usize) * FEATURES + 20] = 1.0;
            x[((t ^ flip) as usize) * FEATURES + 21] = 1.0;
        }
        x
    }

    fn ply(&self) -> usize {
        self.moves.len()
    }

    fn clone_board(&self) -> Board {
        Board {
            stack: self.stack.clone(),
            moves: self.moves.clone(),
            hashes: self.hashes.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Phase C: the batched Gumbel search tree, entirely in Rust. Python's only
// job is evaluating position batches with the net. Semantics mirror
// prophet/searchB.py (the validated spec): Gumbel top-k root, sequential
// halving, PUCT descent with virtual loss, q_trust warm-start via the
// dueling composition, parity draw-contempt.
// ---------------------------------------------------------------------------

const NUM_ACTIONS: usize = 4096;

#[derive(Clone)]
struct SNode {
    prior: f32,
    q_init: f32,
    q_raw: f32, // dueling-composed q BEFORE q_trust (training's q_head signal)
    visits: u32,
    total: f32,
    vloss: u32,
    expanded: bool,
    child_start: u32,
    child_len: u16,
}

impl SNode {
    fn leaf(prior: f32, q_init: f32, q_raw: f32) -> Self {
        SNode {
            prior,
            q_init,
            q_raw,
            visits: 0,
            total: 0.0,
            vloss: 0,
            expanded: false,
            child_start: 0,
            child_len: 0,
        }
    }
}

struct Pending {
    path: Vec<u32>,     // child-node indices, root child first, leaf last
    actions: Vec<u16>,  // actions along the path (same length)
    legal: Vec<u16>,    // leaf's legal actions (for expansion)
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct XorShift(u64);
impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn gumbel(&mut self) -> f64 {
        let u = self.uniform().max(1e-12).min(1.0 - 1e-12);
        -(-(u.ln())).ln()
    }
}

#[pyclass]
pub struct BatchSearch {
    board: Board,
    nodes: Vec<SNode>,
    child_actions: Vec<u16>,
    child_nodes: Vec<u32>,
    budget: u32,
    batch: usize,
    candidates: usize,
    c_puct: f32,
    c_visit: f32,
    c_scale: f32,
    q_trust: f32,
    contempt: f32,
    rng: XorShift,
    // root bookkeeping
    base: Vec<(u16, f64)>, // (action, gumbel base) for root legal moves
    remaining: Vec<u16>,
    spent: u32,
    per_phase: u32,
    phase_spent: u32,
    pending: Vec<Pending>,
    finished: bool,
    rr_cursor: usize, // round-robin arm cursor, persists across collect calls
    // training-stats retention
    root_legal: Vec<u16>,
    root_logits_legal: Vec<f32>,
    root_v_net: f32,
}

impl BatchSearch {
    fn dueling_table(v: f32, legal: &[u16], logits: &[f32], adv: &[f32]) -> Vec<(f32, f32)> {
        // returns (prior, q) per legal action
        let mut mx = f32::NEG_INFINITY;
        for &a in legal {
            mx = mx.max(logits[a as usize]);
        }
        let mut ps: Vec<f32> = legal.iter().map(|&a| (logits[a as usize] - mx).exp()).collect();
        let s: f32 = ps.iter().sum();
        for p in ps.iter_mut() {
            *p /= s;
        }
        let mut amax = f32::NEG_INFINITY;
        for &a in legal {
            amax = amax.max(adv[a as usize]);
        }
        let vc = v.clamp(-0.997, 0.997);
        let base = 0.5 * ((1.0 + vc) / (1.0 - vc)).ln(); // atanh
        legal
            .iter()
            .zip(ps.iter())
            .map(|(&a, &p)| (p, (base + adv[a as usize] - amax).tanh()))
            .collect()
    }

    fn expand_node(&mut self, node_idx: u32, legal: &[u16], logits: &[f32], adv: &[f32], v: f32) {
        let table = Self::dueling_table(v, legal, logits, adv);
        let start = self.nodes.len() as u32;
        for (_a, &(p, q)) in legal.iter().zip(table.iter()) {
            self.nodes.push(SNode::leaf(p, self.q_trust * q, q));
        }
        let cstart = self.child_actions.len() as u32;
        for (k, &a) in legal.iter().enumerate() {
            self.child_actions.push(a);
            self.child_nodes.push(start + k as u32);
        }
        let n = &mut self.nodes[node_idx as usize];
        n.child_start = cstart;
        n.child_len = legal.len() as u16;
        n.expanded = true;
    }

    fn pick_child(&self, node_idx: u32) -> (u16, u32) {
        let n = &self.nodes[node_idx as usize];
        let sqrt_n = (((n.visits + n.vloss).max(1)) as f32).sqrt();
        let mut best = (0u16, 0u32);
        let mut best_s = f32::NEG_INFINITY;
        for k in 0..n.child_len as usize {
            let a = self.child_actions[n.child_start as usize + k];
            let ci = self.child_nodes[n.child_start as usize + k];
            let c = &self.nodes[ci as usize];
            let vis = c.visits + c.vloss;
            let q = if vis > 0 {
                -((c.total - c.vloss as f32) / vis as f32)
            } else {
                c.q_init
            };
            let s = q + self.c_puct * c.prior * sqrt_n / (1.0 + vis as f32);
            if s > best_s {
                best_s = s;
                best = (a, ci);
            }
        }
        best
    }

    fn backup(&mut self, path: &[u32], leaf_value: f32) {
        let mut v = leaf_value;
        for &ci in path.iter().rev() {
            let c = &mut self.nodes[ci as usize];
            c.visits += 1;
            c.total += v;
            if c.vloss > 0 {
                c.vloss -= 1;
            }
            v = -v;
        }
        self.nodes[0].visits += 1;
    }

    fn completed_q(&self, action: u16) -> f32 {
        let n = &self.nodes[0];
        for k in 0..n.child_len as usize {
            if self.child_actions[n.child_start as usize + k] == action {
                let c = &self.nodes[self.child_nodes[n.child_start as usize + k] as usize];
                return if c.visits > 0 {
                    -(c.total / c.visits as f32)
                } else {
                    c.q_init
                };
            }
        }
        -1.0
    }

    fn max_root_child_visits(&self) -> u32 {
        let n = &self.nodes[0];
        let mut mx = 0;
        for k in 0..n.child_len as usize {
            let c = &self.nodes[self.child_nodes[n.child_start as usize + k] as usize];
            mx = mx.max(c.visits);
        }
        mx
    }

    fn halve(&mut self) {
        if self.remaining.len() <= 1 {
            self.finished = self.spent >= self.budget || self.remaining.len() <= 1;
            return;
        }
        let sig = (self.c_visit + self.max_root_child_visits() as f32) * self.c_scale;
        let base = &self.base;
        let mut scored: Vec<(u16, f64)> = self
            .remaining
            .iter()
            .map(|&a| {
                let b = base.iter().find(|(x, _)| *x == a).map(|(_, g)| *g).unwrap_or(-1e9);
                (a, b + (sig * self.completed_q(a)) as f64)
            })
            .collect();
        scored.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        let keep = (scored.len() / 2).max(1);
        self.remaining = scored.into_iter().take(keep).map(|(a, _)| a).collect();
        self.phase_spent = 0;
    }

    fn collect_one(&mut self, first_action: u16) -> Option<Pending> {
        // walk from root taking first_action, then PUCT; returns pending eval
        // job or None if the path hit a terminal (handled inline).
        let n0 = &self.nodes[0];
        let mut ci = u32::MAX;
        for k in 0..n0.child_len as usize {
            if self.child_actions[n0.child_start as usize + k] == first_action {
                ci = self.child_nodes[n0.child_start as usize + k];
                break;
            }
        }
        if ci == u32::MAX {
            return None;
        }
        let mut path = vec![ci];
        let mut actions = vec![first_action];
        let _ = self.board.push_action(first_action);
        self.nodes[ci as usize].vloss += 1;
        loop {
            let cur = path[path.len() - 1];
            if !self.nodes[cur as usize].expanded {
                if let Some(mut term) = self.board.terminal_value() {
                    if term == 0.0 && self.contempt != 0.0 {
                        let d = path.len();
                        term = if d % 2 == 0 { -self.contempt } else { self.contempt };
                    }
                    self.backup(&path, term);
                    for _ in 0..path.len() {
                        self.board.pop();
                    }
                    self.spent += 1;
                    self.phase_spent += 1;
                    return None;
                }
                let legal = self.board.legal_actions();
                for _ in 0..path.len() {
                    self.board.pop();
                }
                return Some(Pending { path, actions, legal });
            }
            let (a, next) = self.pick_child(cur);
            let _ = self.board.push_action(a);
            self.nodes[next as usize].vloss += 1;
            path.push(next);
            actions.push(a);
        }
    }

    // Heavy bodies of collect/apply, GIL-free (called via py.allow_threads so
    // many worker threads can run their trees truly in parallel).
    fn collect_inner(&mut self) -> Vec<u8> {
        self.pending.clear();
        if self.finished || self.spent >= self.budget || self.remaining.is_empty() {
            self.finished = true;
            return Vec::new();
        }
        let want = self
            .batch
            .min((self.budget - self.spent) as usize)
            .min((self.per_phase.saturating_sub(self.phase_spent)).max(1) as usize);
        let mut feats: Vec<f32> = Vec::with_capacity(want * 64 * FEATURES);
        let terminals_before = self.spent;
        let rem = self.remaining.clone();
        // Round-robin over the remaining arms with a PERSISTENT cursor, so
        // sims spread uniformly across candidates no matter how `want`
        // relates to the arm count (a loop restarting at arm 0 each call
        // starves the tail arms whenever want < len(rem) — at batch=1 it
        // starves every arm but the first).
        let mut tries = 0;
        while self.pending.len() < want && self.spent < self.budget && tries < 4 * want + 8 {
            let a = rem[self.rr_cursor % rem.len()];
            self.rr_cursor = self.rr_cursor.wrapping_add(1);
            tries += 1;
            if let Some(p) = self.collect_one(a) {
                // encode leaf: replay path
                for &act in p.actions.iter() {
                    let _ = self.board.push_action(act);
                }
                feats.extend(self.board.encode());
                for _ in 0..p.actions.len() {
                    self.board.pop();
                }
                self.pending.push(p);
            }
        }
        if self.pending.is_empty() {
            if self.spent == terminals_before {
                // nothing collectable at all: degenerate, stop
                self.finished = true;
            }
            if self.phase_spent >= self.per_phase {
                self.halve();
            }
            return Vec::new();
        }
        feats.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn apply_inner(&mut self, logits: &[u8], adv: &[u8], values: &[u8]) {
        let lg = bytes_to_f32(logits);
        let ad = bytes_to_f32(adv);
        let vs = bytes_to_f32(values);
        let jobs = std::mem::take(&mut self.pending);
        for (j, p) in jobs.iter().enumerate() {
            let leaf = p.path[p.path.len() - 1];
            let lo = j * NUM_ACTIONS;
            if !self.nodes[leaf as usize].expanded {
                let lslice = &lg[lo..lo + NUM_ACTIONS];
                let aslice = &ad[lo..lo + NUM_ACTIONS];
                self.expand_node(leaf, &p.legal, lslice, aslice, vs[j]);
            }
            self.backup(&p.path, vs[j]);
        }
        self.spent += jobs.len() as u32;
        self.phase_spent += jobs.len() as u32;
        if self.phase_spent >= self.per_phase {
            self.halve();
        }
        if self.spent >= self.budget {
            self.finished = true;
        }
    }

    fn set_root_inner(&mut self, logits: &[u8], adv: &[u8], v: f32) {
        let lg = bytes_to_f32(logits);
        let ad = bytes_to_f32(adv);
        let legal = self.board.legal_actions();
        self.expand_node(0, &legal, &lg, &ad, v);
        self.nodes[0].visits = 1;
        // retain for training-target export
        self.root_logits_legal = legal.iter().map(|&a| lg[a as usize]).collect();
        self.root_legal = legal.clone();
        self.root_v_net = v;
        let mut base: Vec<(u16, f64)> = legal
            .iter()
            .map(|&a| (a, lg[a as usize] as f64 + self.rng.gumbel()))
            .collect();
        base.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        let m = self.candidates.min(base.len());
        self.remaining = base.iter().take(m).map(|(a, _)| *a).collect();
        self.base = base;
        self.spent = 1;
        let phases = if m > 1 { (m as f32).log2().ceil() as u32 } else { 1 };
        self.per_phase = ((self.budget - 1) / phases.max(1)).max(1);
        self.phase_spent = 0;
    }
}

#[pymethods]
impl BatchSearch {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(fen: &str, budget: u32, batch: usize, candidates: usize, c_puct: f32,
           c_visit: f32, c_scale: f32, q_trust: f32, contempt: f32, seed: u64)
           -> PyResult<Self> {
        let board = Board::from_fen(fen)?;
        Ok(BatchSearch {
            board,
            nodes: vec![SNode::leaf(0.0, 0.0, 0.0)],
            child_actions: vec![],
            child_nodes: vec![],
            budget: budget.max(2),
            batch: batch.max(1),
            candidates: candidates.max(1),
            c_puct,
            c_visit,
            c_scale,
            q_trust,
            contempt,
            rng: XorShift(seed | 1),
            base: vec![],
            remaining: vec![],
            spent: 0,
            per_phase: 1,
            phase_spent: 0,
            pending: vec![],
            finished: false,
            rr_cursor: 0,
            root_legal: vec![],
            root_logits_legal: vec![],
            root_v_net: 0.0,
        })
    }

    fn root_features(&self) -> Vec<f32> {
        self.board.encode()
    }

    fn set_root(&mut self, py: Python<'_>, logits: &[u8], adv: &[u8], v: f32) {
        py.allow_threads(|| self.set_root_inner(logits, adv, v));
    }

    /// Collect the next batch; returns flattened features (empty = done).
    fn collect(&mut self, py: Python<'_>) -> PyObject {
        use pyo3::types::PyBytes;
        let bytes = py.allow_threads(|| self.collect_inner());
        PyBytes::new(py, &bytes).into()
    }

    fn n_pending(&self) -> usize {
        self.pending.len()
    }

    fn apply(&mut self, py: Python<'_>, logits: &[u8], adv: &[u8], values: &[u8]) {
        py.allow_threads(|| self.apply_inner(logits, adv, values));
    }

    fn done(&self) -> bool {
        self.finished || self.spent >= self.budget
    }

    fn spent_forwards(&self) -> u32 {
        self.spent
    }

    fn best(&self) -> u16 {
        let sig = (self.c_visit + self.max_root_child_visits() as f32) * self.c_scale;
        let pool: Vec<u16> = if self.remaining.is_empty() {
            self.base.iter().map(|(a, _)| *a).collect()
        } else {
            self.remaining.clone()
        };
        let mut best = pool[0];
        let mut best_s = f64::NEG_INFINITY;
        for &a in pool.iter() {
            let b = self
                .base
                .iter()
                .find(|(x, _)| *x == a)
                .map(|(_, g)| *g)
                .unwrap_or(-1e9);
            let s = b + (sig * self.completed_q(a)) as f64;
            if s > best_s {
                best_s = s;
                best = a;
            }
        }
        best
    }

    // ---- training-stats exports (Rust-max pipeline) ----

    /// Improved policy target: (actions, probs) = softmax over legal of
    /// (root logit + sig * completedQ). Mirrors search.py's policy target.
    fn policy_target(&self) -> (Vec<u16>, Vec<f32>) {
        let sig = (self.c_visit + self.max_root_child_visits() as f32) * self.c_scale;
        let mut scores: Vec<f32> = self
            .root_legal
            .iter()
            .zip(self.root_logits_legal.iter())
            .map(|(&a, &l)| l + sig * self.completed_q(a))
            .collect();
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in scores.iter_mut() {
            *x = (*x - mx).exp();
            sum += *x;
        }
        for x in scores.iter_mut() {
            *x /= sum;
        }
        (self.root_legal.clone(), scores)
    }

    /// Root children that got visits: (actions, empirical q from the root
    /// player's perspective, visit counts). Q-head targets.
    fn visited_children(&self) -> (Vec<u16>, Vec<f32>, Vec<u32>) {
        let n = &self.nodes[0];
        let mut acts = Vec::new();
        let mut qs = Vec::new();
        let mut vis = Vec::new();
        for k in 0..n.child_len as usize {
            let a = self.child_actions[n.child_start as usize + k];
            let c = &self.nodes[self.child_nodes[n.child_start as usize + k] as usize];
            if c.visits > 0 {
                acts.push(a);
                qs.push(-(c.total / c.visits as f32));
                vis.push(c.visits);
            }
        }
        (acts, qs, vis)
    }

    /// Search-blended root value: (v_net + n * q_avg) / (1 + n), where n is
    /// total root-child visits and q_avg their visit-weighted mean q.
    fn root_value(&self) -> f32 {
        let n = &self.nodes[0];
        let mut tot = 0.0f32;
        let mut cnt = 0u32;
        for k in 0..n.child_len as usize {
            let c = &self.nodes[self.child_nodes[n.child_start as usize + k] as usize];
            if c.visits > 0 {
                tot += -c.total; // child perspective -> root perspective
                cnt += c.visits;
            }
        }
        if cnt == 0 {
            return self.root_v_net;
        }
        (self.root_v_net + tot) / (1.0 + cnt as f32)
    }

    /// Net's raw root value (pre-search).
    fn net_v(&self) -> f32 {
        self.root_v_net
    }

    /// Raw dueling-composed q (pre-q_trust) of a root action — the q-head's
    /// own opinion of the move, used as q_head_played in training records.
    fn root_q_raw(&self, action: u16) -> f32 {
        let n = &self.nodes[0];
        for k in 0..n.child_len as usize {
            if self.child_actions[n.child_start as usize + k] == action {
                return self.nodes[self.child_nodes[n.child_start as usize + k] as usize].q_raw;
            }
        }
        0.0
    }
}

#[pymodule]
fn prophet_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Board>()?;
    m.add_class::<BatchSearch>()?;
    let _ = (Rank::First, Color::White); // keep imports used across versions
    Ok(())
}
