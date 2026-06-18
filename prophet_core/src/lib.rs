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

#[pymodule]
fn prophet_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Board>()?;
    let _ = (Rank::First, Color::White); // keep imports used across versions
    Ok(())
}
