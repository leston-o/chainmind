//! Bit-reproducible math. Only ops that IEEE 754 defines exactly
//! (+, -, *, /, sqrt, floor) are used, never libm.

const LOG2E: f32 = 1.442_695_f32;
const LN2_HI: f32 = 0.693_145_751_953_125;
const LN2_LO: f32 = 1.428_606_820_309_417_2e-6;

/// e^x, accurate to a few ulp, identical on every platform.
#[inline]
pub fn expf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > 88.0 {
        return f32::INFINITY;
    }
    if x < -87.0 {
        return 0.0;
    }
    let n = (x * LOG2E + 0.5).floor();
    let r = (x - n * LN2_HI) - n * LN2_LO;
    // Taylor to r^7 on |r| <= ln2/2.
    let mut p = 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    p * scale
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + expf(-x))
}

pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32]) {
    let mut ss = 0.0f32;
    for v in x {
        ss += v * v;
    }
    ss = ss / x.len() as f32 + 1e-5;
    let inv = 1.0 / ss.sqrt();
    for i in 0..x.len() {
        out[i] = w[i] * (inv * x[i]);
    }
}

pub fn softmax(x: &mut [f32]) {
    let mut max = x[0];
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = expf(*v - max);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Round half away from zero, without libm.
#[inline]
pub fn round_i32(v: f32) -> i32 {
    if v >= 0.0 {
        (v + 0.5).floor() as i32
    } else {
        -((-v + 0.5).floor() as i32)
    }
}

/// Symmetric int8 quantization of a whole vector. Returns the scale.
pub fn quantize(out: &mut [i8], x: &[f32]) -> f32 {
    let mut amax = 0.0f32;
    for &v in x {
        let a = if v < 0.0 { -v } else { v };
        if a > amax {
            amax = a;
        }
    }
    if amax == 0.0 {
        for q in out.iter_mut() {
            *q = 0;
        }
        return 0.0;
    }
    let scale = amax / 127.0;
    let inv = 1.0 / scale;
    for i in 0..x.len() {
        out[i] = round_i32(x[i] * inv).clamp(-127, 127) as i8;
    }
    scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expf_matches_std_closely() {
        let mut x = -20.0f32;
        while x < 20.0 {
            let got = expf(x) as f64;
            let want = (x as f64).exp();
            assert!(((got - want) / want).abs() < 2e-6, "x={x} got={got} want={want}");
            x += 0.0137;
        }
        assert_eq!(expf(0.0), 1.0);
        assert_eq!(expf(-100.0), 0.0);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, -4.0];
        softmax(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(v[2] > v[1] && v[1] > v[0]);
    }

    #[test]
    fn quantize_roundtrip() {
        let x = [0.5f32, -1.0, 0.25, 0.0];
        let mut q = [0i8; 4];
        let s = quantize(&mut q, &x);
        assert_eq!(q, [64, -127, 32, 0]);
        assert!((q[0] as f32 * s - 0.5).abs() < 0.01);
    }
}
