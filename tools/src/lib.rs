//! Writer for the `CMND` v1 model blob (the reader lives in the engine).

use chainmind_engine::model::{MAGIC, VERSION};
use chainmind_engine::Config;

pub struct FloatLayer {
    pub rms_att: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub rms_ffn: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub w3: Vec<f32>,
}

/// A model in plain f32, as read from a checkpoint.
pub struct FloatModel {
    pub config: Config,
    pub tok_emb: Vec<f32>,
    pub layers: Vec<FloatLayer>,
    pub rms_final: Vec<f32>,
    pub wcls: Option<Vec<f32>>,
    pub vocab: Vec<Vec<u8>>,
    pub scores: Vec<f32>,
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_f32s(out: &mut Vec<u8>, v: &[f32]) {
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// int8 weights, one f32 scale per row.
fn put_qmat(out: &mut Vec<u8>, w: &[f32], rows: usize, cols: usize) {
    assert_eq!(w.len(), rows * cols, "matrix size mismatch");
    let mut scales = Vec::with_capacity(rows);
    for row in w.chunks_exact(cols) {
        let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        for &v in row {
            let q = if scale == 0.0 { 0.0 } else { (v / scale).round() };
            out.push(q.clamp(-127.0, 127.0) as i8 as u8);
        }
        scales.push(scale);
    }
    put_f32s(out, &scales);
}

impl FloatModel {
    pub fn to_blob(&self) -> Vec<u8> {
        let c = self.config;
        let (dim, hidden, kv_dim) = (c.dim as usize, c.hidden_dim as usize, c.kv_dim());
        let vocab = c.vocab_size as usize;
        let hs = c.head_size();

        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        put_u32(&mut out, VERSION);
        for v in [
            c.dim,
            c.hidden_dim,
            c.n_layers,
            c.n_heads,
            c.n_kv_heads,
            c.vocab_size,
            c.seq_len,
        ] {
            put_u32(&mut out, v);
        }
        put_u32(&mut out, if self.wcls.is_none() { 1 } else { 0 });

        put_qmat(&mut out, &self.tok_emb, vocab, dim);
        for l in &self.layers {
            put_f32s(&mut out, &l.rms_att);
            put_qmat(&mut out, &l.wq, dim, dim);
            put_qmat(&mut out, &l.wk, kv_dim, dim);
            put_qmat(&mut out, &l.wv, kv_dim, dim);
            put_qmat(&mut out, &l.wo, dim, dim);
            put_f32s(&mut out, &l.rms_ffn);
            put_qmat(&mut out, &l.w1, hidden, dim);
            put_qmat(&mut out, &l.w2, dim, hidden);
            put_qmat(&mut out, &l.w3, hidden, dim);
        }
        put_f32s(&mut out, &self.rms_final);

        // RoPE tables, computed here in f64 so the chain never needs trig.
        let half = hs / 2;
        let mut cos = Vec::with_capacity(c.seq_len as usize * half);
        let mut sin = Vec::with_capacity(c.seq_len as usize * half);
        for pos in 0..c.seq_len as usize {
            for j in 0..half {
                let freq = 1.0 / 10000f64.powf((2 * j) as f64 / hs as f64);
                let v = pos as f64 * freq;
                cos.push(v.cos() as f32);
                sin.push(v.sin() as f32);
            }
        }
        put_f32s(&mut out, &cos);
        put_f32s(&mut out, &sin);

        if let Some(w) = &self.wcls {
            put_qmat(&mut out, w, vocab, dim);
        }

        let max_len = self.vocab.iter().map(|p| p.len()).max().unwrap_or(0) as u32;
        put_u32(&mut out, max_len);
        for (piece, score) in self.vocab.iter().zip(&self.scores) {
            out.extend_from_slice(&score.to_le_bytes());
            put_u32(&mut out, piece.len() as u32);
            out.extend_from_slice(piece);
        }
        out
    }
}

/// Read a legacy llama2.c checkpoint (`.bin`, version 0) and its tokenizer.
pub fn read_llama2c(checkpoint: &[u8], tokenizer: &[u8]) -> Result<FloatModel, String> {
    let mut off = 0usize;
    let i32_at =|off: &mut usize| -> Result<i32, String> {
        let b = checkpoint.get(*off..*off + 4).ok_or("checkpoint truncated")?;
        *off += 4;
        Ok(i32::from_le_bytes(b.try_into().unwrap()))
    };
    let dim = i32_at(&mut off)? as u32;
    let hidden_dim = i32_at(&mut off)? as u32;
    let n_layers = i32_at(&mut off)? as u32;
    let n_heads = i32_at(&mut off)? as u32;
    let n_kv_heads = i32_at(&mut off)? as u32;
    let vocab_raw = i32_at(&mut off)?;
    let seq_len = i32_at(&mut off)? as u32;
    let shared = vocab_raw > 0;
    let config = Config {
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size: vocab_raw.unsigned_abs(),
        seq_len,
    };
    let floats: Vec<f32> = checkpoint[off..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut p = 0usize;
    let mut take = |n: usize| -> Result<Vec<f32>, String> {
        let s = floats.get(p..p + n).ok_or("checkpoint truncated")?.to_vec();
        p += n;
        Ok(s)
    };
    let (d, h, l, kv) = (
        dim as usize,
        hidden_dim as usize,
        n_layers as usize,
        config.kv_dim(),
    );
    let v = config.vocab_size as usize;
    let hs = config.head_size();

    // llama2.c stores each weight kind for all layers contiguously.
    let tok_emb = take(v * d)?;
    let rms_att = take(l * d)?;
    let wq = take(l * d * d)?;
    let wk = take(l * d * kv)?;
    let wv = take(l * d * kv)?;
    let wo = take(l * d * d)?;
    let rms_ffn = take(l * d)?;
    let w1 = take(l * h * d)?;
    let w2 = take(l * d * h)?;
    let w3 = take(l * h * d)?;
    let rms_final = take(d)?;
    // Skip the stored freq_cis tables; we compute our own.
    take(seq_len as usize * hs / 2)?;
    take(seq_len as usize * hs / 2)?;
    let wcls = if shared { None } else { Some(take(v * d)?) };

    let slice = |w: &[f32], i: usize, n: usize| w[i * n..(i + 1) * n].to_vec();
    let layers = (0..l)
        .map(|i| FloatLayer {
            rms_att: slice(&rms_att, i, d),
            wq: slice(&wq, i, d * d),
            wk: slice(&wk, i, d * kv),
            wv: slice(&wv, i, d * kv),
            wo: slice(&wo, i, d * d),
            rms_ffn: slice(&rms_ffn, i, d),
            w1: slice(&w1, i, h * d),
            w2: slice(&w2, i, d * h),
            w3: slice(&w3, i, h * d),
        })
        .collect();

    // Tokenizer: max_len i32, then per token: f32 score, i32 len, bytes.
    let mut t = 4usize;
    let mut vocab = Vec::with_capacity(v);
    let mut scores = Vec::with_capacity(v);
    for _ in 0..v {
        let rd = |a: usize| tokenizer.get(a..a + 4).ok_or("tokenizer truncated");
        scores.push(f32::from_le_bytes(rd(t)?.try_into().unwrap()));
        let len = i32::from_le_bytes(rd(t + 4)?.try_into().unwrap()) as usize;
        t += 8;
        vocab.push(
            tokenizer
                .get(t..t + len)
                .ok_or("tokenizer truncated")?
                .to_vec(),
        );
        t += len;
    }

    Ok(FloatModel {
        config,
        tok_emb,
        layers,
        rms_final,
        wcls,
        vocab,
        scores,
    })
}

/// A model with random weights but a real shape, for gas benchmarks and tests.
pub fn random_model(config: Config, seed: u64) -> FloatModel {
    let mut s = seed | 1;
    let mut rnd = move |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    };
    let (d, h, kv) = (
        config.dim as usize,
        config.hidden_dim as usize,
        config.kv_dim(),
    );
    let v = config.vocab_size as usize;
    let layers = (0..config.n_layers)
        .map(|_| FloatLayer {
            rms_att: vec![1.0; d],
            wq: rnd(d * d, 0.2),
            wk: rnd(kv * d, 0.2),
            wv: rnd(kv * d, 0.2),
            wo: rnd(d * d, 0.2),
            rms_ffn: vec![1.0; d],
            w1: rnd(h * d, 0.2),
            w2: rnd(d * h, 0.2),
            w3: rnd(h * d, 0.2),
        })
        .collect();

    // <unk> <s> </s>, 256 byte tokens, then printable chars and pairs.
    let mut vocab: Vec<Vec<u8>> = vec![b"<unk>".to_vec(), b"<s>".to_vec(), b"</s>".to_vec()];
    for b in 0..=255u8 {
        vocab.push(format!("<0x{b:02X}>").into_bytes());
    }
    let printable: Vec<u8> = (b' '..=b'~').collect();
    for &a in &printable {
        vocab.push(vec![a]);
    }
    let mut pair = 0usize;
    while vocab.len() < v && pair < printable.len() * printable.len() {
        let a = printable[pair / printable.len() % printable.len()];
        let b = printable[pair % printable.len()];
        let piece = vec![b' ', a, b];
        if !vocab.contains(&piece) {
            vocab.push(piece);
        }
        pair += 1;
    }
    vocab.truncate(v);
    let scores = (0..v).map(|i| -(i as f32)).collect();

    FloatModel {
        config,
        tok_emb: rnd(v * d, 1.0),
        layers,
        rms_final: vec![1.0; d],
        wcls: None,
        vocab,
        scores,
    }
}

pub fn sha256_hex(b: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// The stories260K shape from llama2.c.
pub fn stories260k_config() -> Config {
    Config {
        dim: 64,
        hidden_dim: 172,
        n_layers: 5,
        n_heads: 8,
        n_kv_heads: 4,
        vocab_size: 512,
        seq_len: 512,
    }
}
