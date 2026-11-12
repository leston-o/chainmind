//! SentencePiece-style BPE (llama2.c compatible), borrowing the blob.

use crate::model::{ParseError, Reader};
use crate::BOS;

pub struct Tokenizer<'a> {
    vocab: Vec<&'a [u8]>,
    scores: Vec<f32>,
    /// Token ids sorted by their bytes, for binary search.
    sorted: Vec<u32>,
}

impl<'a> Tokenizer<'a> {
    pub(crate) fn parse(r: &mut Reader<'a>, vocab_size: usize) -> Result<Self, ParseError> {
        let max_len = r.u32()? as usize;
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            scores.push(r.f32()?);
            let len = r.u32()? as usize;
            if len > max_len {
                return Err(ParseError::BadConfig);
            }
            vocab.push(r.bytes(len)?);
        }
        let mut sorted: Vec<u32> = (0..vocab_size as u32).collect();
        sorted.sort_by(|&a, &b| vocab[a as usize].cmp(vocab[b as usize]).then(a.cmp(&b)));
        Ok(Self {
            vocab,
            scores,
            sorted,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn lookup(&self, piece: &[u8]) -> Option<u32> {
        self.sorted
            .binary_search_by(|&id| self.vocab[id as usize].cmp(piece))
            .ok()
            .map(|i| self.sorted[i])
    }

    pub fn encode(&self, text: &str, bos: bool) -> Vec<u32> {
        let mut tokens: Vec<u32> = Vec::with_capacity(text.len() + 2);
        if bos {
            tokens.push(BOS);
        }
        if !text.is_empty() {
            // SentencePiece adds a dummy space prefix.
            if let Some(id) = self.lookup(b" ") {
                tokens.push(id);
            }
        }
        let mut buf = [0u8; 4];
        for ch in text.chars() {
            let bytes = ch.encode_utf8(&mut buf).as_bytes();
            match self.lookup(bytes) {
                Some(id) => tokens.push(id),
                // Byte fallback: <0x00>.. live at ids 3..259.
                None => tokens.extend(bytes.iter().map(|&b| b as u32 + 3)),
            }
        }
        if tokens.is_empty() {
            tokens.push(BOS);
        }

        // Greedily merge the best-scoring adjacent pair until none merge.
        let mut joined: Vec<u8> = Vec::with_capacity(64);
        loop {
            let mut best: Option<(f32, u32, usize)> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                joined.clear();
                joined.extend_from_slice(self.vocab[tokens[i] as usize]);
                joined.extend_from_slice(self.vocab[tokens[i + 1] as usize]);
                if let Some(id) = self.lookup(&joined) {
                    let score = self.scores[id as usize];
                    if best.map_or(true, |(s, _, _)| score > s) {
                        best = Some((score, id, i));
                    }
                }
            }
            match best {
                Some((_, id, i)) => {
                    tokens[i] = id;
                    tokens.remove(i + 1);
                }
                None => break,
            }
        }
        tokens
    }

    /// Bytes for `token`, given the token before it.
    pub fn decode(&self, prev: u32, token: u32) -> Vec<u8> {
        let mut piece = self.vocab[token as usize];
        if prev == BOS && piece.first() == Some(&b' ') {
            piece = &piece[1..];
        }
        // Raw byte tokens look like "<0x0A>".
        if piece.len() == 6 && piece.starts_with(b"<0x") && piece[5] == b'>' {
            if let Ok(s) = core::str::from_utf8(&piece[3..5]) {
                if let Ok(b) = u8::from_str_radix(s, 16) {
                    return vec![b];
                }
            }
        }
        piece.to_vec()
    }
}
