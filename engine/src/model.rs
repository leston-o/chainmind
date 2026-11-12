//! Model blob parsing and the transformer forward pass.

use crate::math::{quantize, rmsnorm, silu, softmax};
use crate::tokenizer::Tokenizer;

pub const MAGIC: &[u8; 4] = b"CMND";
pub const VERSION: u32 = 1;
const FLAG_SHARED_CLASSIFIER: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub dim: u32,
    pub hidden_dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub vocab_size: u32,
    pub seq_len: u32,
}

impl Config {
    pub fn head_size(&self) -> usize {
        (self.dim / self.n_heads) as usize
    }
    pub fn kv_dim(&self) -> usize {
        (self.dim * self.n_kv_heads / self.n_heads) as usize
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    BadMagic,
    BadVersion(u32),
    BadConfig,
    Truncated,
    TrailingBytes,
}

/// Row-major int8 matrix with one f32 scale per row. Borrows the blob.
pub struct QMatrix<'a> {
    pub rows: usize,
    pub cols: usize,
    q: &'a [i8],
    s: Vec<f32>,
}

impl<'a> QMatrix<'a> {
    /// out = W · x, where x is int8 with scale `xs`.
    #[inline]
    pub fn matmul(&self, out: &mut [f32], x: &[i8], xs: f32) {
        let x = &x[..self.cols];
        for (i, (row, ws)) in self.q.chunks_exact(self.cols).zip(&self.s).enumerate() {
            let acc: i32 = row
                .iter()
                .zip(x)
                .map(|(&a, &b)| a as i32 * b as i32)
                .sum();
            out[i] = acc as f32 * (ws * xs);
        }
    }

    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        let row = &self.q[r * self.cols..(r + 1) * self.cols];
        let s = self.s[r];
        for (o, &q) in out.iter_mut().zip(row) {
            *o = q as f32 * s;
        }
    }
}

pub struct Layer<'a> {
    rms_att: Vec<f32>,
    wq: QMatrix<'a>,
    wk: QMatrix<'a>,
    wv: QMatrix<'a>,
    wo: QMatrix<'a>,
    rms_ffn: Vec<f32>,
    w1: QMatrix<'a>,
    w2: QMatrix<'a>,
    w3: QMatrix<'a>,
}

pub struct Model<'a> {
    pub config: Config,
    pub tokenizer: Tokenizer<'a>,
    tok_emb: QMatrix<'a>,
    layers: Vec<Layer<'a>>,
    rms_final: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    wcls: Option<QMatrix<'a>>,
}

/// Scratch buffers and the KV cache for one generation.
pub struct State {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    xq: Vec<i8>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    max_pos: usize,
}

pub(crate) struct Reader<'a> {
    b: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(b: &'a [u8]) -> Self {
        Self { b, off: 0 }
    }
    pub(crate) fn bytes(&mut self, n: usize) -> Result<&'a [u8], ParseError> {
        let end = self.off.checked_add(n).ok_or(ParseError::Truncated)?;
        let s = self.b.get(self.off..end).ok_or(ParseError::Truncated)?;
        self.off = end;
        Ok(s)
    }
    pub(crate) fn u32(&mut self) -> Result<u32, ParseError> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    pub(crate) fn f32(&mut self) -> Result<f32, ParseError> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, ParseError> {
        let raw = self.bytes(n * 4)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
    fn i8s(&mut self, n: usize) -> Result<&'a [i8], ParseError> {
        let raw = self.bytes(n)?;
        // SAFETY: u8 and i8 have the same size and alignment; every bit
        // pattern is valid for both.
        Ok(unsafe { core::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) })
    }
    fn qmat(&mut self, rows: usize, cols: usize) -> Result<QMatrix<'a>, ParseError> {
        let q = self.i8s(rows * cols)?;
        let s = self.f32s(rows)?;
        Ok(QMatrix { rows, cols, q, s })
    }
    pub(crate) fn done(&self) -> bool {
        self.off == self.b.len()
    }
}

impl<'a> Model<'a> {
    pub fn parse(blob: &'a [u8]) -> Result<Self, ParseError> {
        let mut r = Reader::new(blob);
        if r.bytes(4)? != MAGIC {
            return Err(ParseError::BadMagic);
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(ParseError::BadVersion(version));
        }
        let config = Config {
            dim: r.u32()?,
            hidden_dim: r.u32()?,
            n_layers: r.u32()?,
            n_heads: r.u32()?,
            n_kv_heads: r.u32()?,
            vocab_size: r.u32()?,
            seq_len: r.u32()?,
        };
        let flags = r.u32()?;
        let c = config;
        let ok = c.dim > 0
            && c.n_heads > 0
            && c.n_kv_heads > 0
            && c.dim % c.n_heads == 0
            && c.n_heads % c.n_kv_heads == 0
            && c.head_size() % 2 == 0
            && c.vocab_size > 3
            && c.seq_len > 0
            && c.dim <= 4096
            && c.hidden_dim <= 16384
            && c.n_layers <= 64
            && c.vocab_size <= 65536
            && c.seq_len <= 4096;
        if !ok {
            return Err(ParseError::BadConfig);
        }
        let (dim, hidden, kv_dim) = (c.dim as usize, c.hidden_dim as usize, c.kv_dim());
        let vocab = c.vocab_size as usize;

        let tok_emb = r.qmat(vocab, dim)?;
        let mut layers = Vec::with_capacity(c.n_layers as usize);
        for _ in 0..c.n_layers {
            layers.push(Layer {
                rms_att: r.f32s(dim)?,
                wq: r.qmat(dim, dim)?,
                wk: r.qmat(kv_dim, dim)?,
                wv: r.qmat(kv_dim, dim)?,
                wo: r.qmat(dim, dim)?,
                rms_ffn: r.f32s(dim)?,
                w1: r.qmat(hidden, dim)?,
                w2: r.qmat(dim, hidden)?,
                w3: r.qmat(hidden, dim)?,
            });
        }
        let rms_final = r.f32s(dim)?;
        let rope_n = c.seq_len as usize * c.head_size() / 2;
        let rope_cos = r.f32s(rope_n)?;
        let rope_sin = r.f32s(rope_n)?;
        let wcls = if flags & FLAG_SHARED_CLASSIFIER != 0 {
            None
        } else {
            Some(r.qmat(vocab, dim)?)
        };
        let tokenizer = Tokenizer::parse(&mut r, vocab)?;
        if !r.done() {
            return Err(ParseError::TrailingBytes);
        }
        Ok(Self {
            config,
            tokenizer,
            tok_emb,
            layers,
            rms_final,
            rope_cos,
            rope_sin,
            wcls,
        })
    }

    pub fn new_state(&self, max_pos: usize) -> State {
        let c = &self.config;
        let (dim, hidden, kv_dim) = (c.dim as usize, c.hidden_dim as usize, c.kv_dim());
        let max_pos = max_pos.clamp(1, c.seq_len as usize);
        let cache = c.n_layers as usize * max_pos * kv_dim;
        State {
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            xb2: vec![0.0; dim],
            hb: vec![0.0; hidden],
            hb2: vec![0.0; hidden],
            q: vec![0.0; dim],
            att: vec![0.0; max_pos],
            logits: vec![0.0; c.vocab_size as usize],
            xq: vec![0; dim.max(hidden)],
            key_cache: vec![0.0; cache],
            value_cache: vec![0.0; cache],
            max_pos,
        }
    }

    /// One transformer step. Returns logits, or an empty slice when
    /// `want_logits` is false (prefill).
    pub fn forward<'s>(
        &self,
        s: &'s mut State,
        token: u32,
        pos: u32,
        want_logits: bool,
    ) -> &'s mut [f32] {
        let c = &self.config;
        let dim = c.dim as usize;
        let hidden = c.hidden_dim as usize;
        let kv_dim = c.kv_dim();
        let hs = c.head_size();
        let half = hs / 2;
        let kv_mul = (c.n_heads / c.n_kv_heads) as usize;
        let p = pos as usize;
        assert!(p < s.max_pos, "position beyond state");
        let inv_sqrt_hs = 1.0 / (hs as f32).sqrt();

        self.tok_emb
            .dequant_row((token as usize).min(c.vocab_size as usize - 1), &mut s.x);

        for (l, lw) in self.layers.iter().enumerate() {
            let loff = l * s.max_pos * kv_dim;
            let slot = loff + p * kv_dim;

            // Attention.
            rmsnorm(&mut s.xb, &s.x, &lw.rms_att);
            let sx = quantize(&mut s.xq[..dim], &s.xb);
            lw.wq.matmul(&mut s.q, &s.xq, sx);
            lw.wk.matmul(&mut s.key_cache[slot..slot + kv_dim], &s.xq, sx);
            lw.wv.matmul(&mut s.value_cache[slot..slot + kv_dim], &s.xq, sx);

            // RoPE, from the precomputed tables.
            let k = &mut s.key_cache[slot..slot + kv_dim];
            let mut i = 0;
            while i < dim {
                let f = p * half + (i % hs) / 2;
                let (fc, fs) = (self.rope_cos[f], self.rope_sin[f]);
                let (a, b) = (s.q[i], s.q[i + 1]);
                s.q[i] = a * fc - b * fs;
                s.q[i + 1] = a * fs + b * fc;
                if i < kv_dim {
                    let (a, b) = (k[i], k[i + 1]);
                    k[i] = a * fc - b * fs;
                    k[i + 1] = a * fs + b * fc;
                }
                i += 2;
            }

            for h in 0..c.n_heads as usize {
                let qh = &s.q[h * hs..(h + 1) * hs];
                let kv_off = (h / kv_mul) * hs;
                let att = &mut s.att[..=p];
                for (t, a) in att.iter_mut().enumerate() {
                    let base = loff + t * kv_dim + kv_off;
                    let kt = &s.key_cache[base..base + hs];
                    let mut score = 0.0f32;
                    for (x, y) in qh.iter().zip(kt) {
                        score += x * y;
                    }
                    *a = score * inv_sqrt_hs;
                }
                softmax(att);
                let out = &mut s.xb[h * hs..(h + 1) * hs];
                out.iter_mut().for_each(|v| *v = 0.0);
                for (t, &a) in att.iter().enumerate() {
                    let base = loff + t * kv_dim + kv_off;
                    let vt = &s.value_cache[base..base + hs];
                    for (o, v) in out.iter_mut().zip(vt) {
                        *o += a * v;
                    }
                }
            }
            let sx = quantize(&mut s.xq[..dim], &s.xb);
            lw.wo.matmul(&mut s.xb2, &s.xq, sx);
            for (x, d) in s.x.iter_mut().zip(&s.xb2) {
                *x += d;
            }

            // Feed-forward (SwiGLU).
            rmsnorm(&mut s.xb, &s.x, &lw.rms_ffn);
            let sx = quantize(&mut s.xq[..dim], &s.xb);
            lw.w1.matmul(&mut s.hb, &s.xq, sx);
            lw.w3.matmul(&mut s.hb2, &s.xq, sx);
            for (a, b) in s.hb.iter_mut().zip(&s.hb2) {
                *a = silu(*a) * b;
            }
            let sh = quantize(&mut s.xq[..hidden], &s.hb);
            lw.w2.matmul(&mut s.xb, &s.xq, sh);
            for (x, d) in s.x.iter_mut().zip(&s.xb) {
                *x += d;
            }
        }

        if !want_logits {
            return &mut s.logits[..0];
        }
        rmsnorm(&mut s.xb, &s.x, &self.rms_final);
        let sx = quantize(&mut s.xq[..dim], &s.xb);
        self.wcls
            .as_ref()
            .unwrap_or(&self.tok_emb)
            .matmul(&mut s.logits, &s.xq, sx);
        &mut s.logits
    }
}
