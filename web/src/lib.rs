//! Browser build of the engine. No wasm-bindgen: a few C-ABI exports.
//!
//! JS calls `alloc`, copies the model blob in, calls `load`, then
//! `generate`, which returns a pointer to:
//!   u32 n_tokens, u32 prompt_tokens, u32 stop, then per token: u32 id, u32 len, bytes

use chainmind_engine::{GenParams, Model, StopReason};
use std::cell::RefCell;

thread_local! {
    static BLOB: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static OUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Take ownership of `len` bytes at `ptr` as the model. Returns 1 if it parses.
#[no_mangle]
pub extern "C" fn load(ptr: *mut u8, len: usize) -> i32 {
    // SAFETY: ptr came from `alloc(len)` and JS filled all `len` bytes.
    let blob = unsafe { Vec::from_raw_parts(ptr, len, len) };
    let ok = Model::parse(&blob).is_ok();
    BLOB.with(|b| *b.borrow_mut() = blob);
    ok as i32
}

fn put(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[no_mangle]
pub extern "C" fn generate(
    prompt_ptr: *mut u8,
    prompt_len: usize,
    max_tokens: u32,
    temperature: f32,
    seed_lo: u32,
    seed_hi: u32,
) -> *const u8 {
    // SAFETY: as in `load`.
    let prompt_bytes = unsafe { Vec::from_raw_parts(prompt_ptr, prompt_len, prompt_len) };
    let prompt = String::from_utf8_lossy(&prompt_bytes).into_owned();
    let seed = (seed_hi as u64) << 32 | seed_lo as u64;

    BLOB.with(|b| {
        let blob = b.borrow();
        let model = Model::parse(&blob).expect("load a model first");
        let params = GenParams {
            max_new_tokens: max_tokens,
            temperature,
            seed,
        };
        let g = model.generate(&prompt, &params, |_| true);
        let prompt_ids = model.tokenizer.encode(&prompt, true);

        OUT.with(|o| {
            let mut out = o.borrow_mut();
            out.clear();
            put(&mut out, g.token_ids.len() as u32);
            put(&mut out, g.prompt_tokens);
            put(
                &mut out,
                match g.stop {
                    StopReason::EndOfText => 0,
                    StopReason::MaxTokens => 1,
                    StopReason::ContextFull => 2,
                    StopReason::Budget => 3,
                },
            );
            let mut prev = *prompt_ids.last().unwrap();
            for &id in &g.token_ids {
                let piece = model.tokenizer.decode(prev, id);
                put(&mut out, id);
                put(&mut out, piece.len() as u32);
                out.extend_from_slice(&piece);
                prev = id;
            }
            out.as_ptr()
        })
    })
}
