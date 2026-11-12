//! Seeded sampling. Same seed, same logits → same token, everywhere.

use crate::math::softmax;

/// xorshift64* — tiny, fast, fully specified.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // splitmix64 so every seed bit matters; never the all-zero state.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Self(if z == 0 { 1 } else { z })
    }
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }
}

pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

pub fn sample(logits: &mut [f32], temperature: f32, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let inv = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv;
    }
    softmax(logits);
    let coin = rng.next_f32();
    let mut cdf = 0.0f32;
    for (i, p) in logits.iter().enumerate() {
        cdf += p;
        if coin < cdf {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_reproducible() {
        let a: Vec<u32> = {
            let mut r = Rng::new(42);
            (0..5).map(|_| r.next_u32()).collect()
        };
        let b: Vec<u32> = {
            let mut r = Rng::new(42);
            (0..5).map(|_| r.next_u32()).collect()
        };
        assert_eq!(a, b);
        assert_ne!(a, {
            let mut r = Rng::new(43);
            (0..5).map(|_| r.next_u32()).collect::<Vec<_>>()
        });
    }

    #[test]
    fn greedy_picks_max() {
        let mut l = vec![0.1, 3.0, -1.0];
        assert_eq!(sample(&mut l, 0.0, &mut Rng::new(1)), 1);
    }
}
