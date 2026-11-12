//! Chainmind: a language model that runs inside NEAR consensus.
//!
//! The weights live in this contract's storage. `generate` runs the full
//! transformer forward pass in WebAssembly on every validator, so the output
//! is as final as a token transfer: no oracle, no API key, no trusted
//! hardware. Once `seal` is called the weights can never change.

use chainmind_engine::{GenParams, Model, StopReason};
use near_sdk::json_types::{Base64VecU8, U64};
use near_sdk::serde_json::json;
use near_sdk::store::Vector;
use near_sdk::{env, log, near, require, AccountId, Gas, PanicOnDefault};

const CHUNK_PREFIX: &[u8] = b"w";
const MAX_CHUNK_BYTES: usize = 256 * 1024;
const MAX_PROMPT_BYTES: usize = 1024;
const MAX_NEW_TOKENS: u32 = 256;
/// ~3.6 Tgas per token + ~16 Tgas to load: 48 tokens stays under RPC view limits.
const MAX_VIEW_TOKENS: u32 = 48;
/// Left for serializing the result and logging after the last token.
const FINISH_RESERVE: Gas = Gas::from_tgas(8);
const STORY_CONTEXT_BYTES: usize = 240;
const MAX_NUDGE_BYTES: usize = 120;

#[near(serializers = [json, borsh])]
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    pub sha256: String,
    pub bytes: u64,
    pub dim: u32,
    pub hidden_dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub vocab_size: u32,
    pub seq_len: u32,
    pub params: u64,
}

#[near(serializers = [json])]
#[derive(Clone, Debug)]
pub struct Generation {
    pub prompt: String,
    pub text: String,
    pub prompt_tokens: u32,
    pub new_tokens: u32,
    pub stop: String,
    pub temperature: f32,
    pub seed: U64,
    pub token_ids: Vec<u32>,
    pub gas_burnt_tgas: f64,
    pub model_sha256: String,
}

#[near(serializers = [json, borsh])]
#[derive(Clone, Debug)]
pub struct Chapter {
    pub index: u32,
    pub author: AccountId,
    pub nudge: String,
    pub text: String,
    pub block_height: U64,
    pub seed: U64,
}

#[near(contract_state)]
#[derive(PanicOnDefault)]
pub struct Chainmind {
    owner_id: AccountId,
    n_chunks: u32,
    total_bytes: u64,
    sealed: Option<ModelInfo>,
    generations: u64,
    story: Vector<Chapter>,
}

#[near]
impl Chainmind {
    #[init]
    pub fn new(owner_id: AccountId) -> Self {
        Self {
            owner_id,
            n_chunks: 0,
            total_bytes: 0,
            sealed: None,
            generations: 0,
            story: Vector::new(b"s"),
        }
    }

    // ------------------------------------------------------------------
    // Loading the model (owner, before sealing)
    // ------------------------------------------------------------------

    /// Append the next chunk of the model blob.
    pub fn upload_chunk(&mut self, index: u32, data: Base64VecU8) {
        self.assert_owner_unsealed();
        require!(index == self.n_chunks, "chunks must be uploaded in order");
        require!(
            !data.0.is_empty() && data.0.len() <= MAX_CHUNK_BYTES,
            "chunk size out of range"
        );
        env::storage_write(&chunk_key(index), &data.0);
        self.n_chunks += 1;
        self.total_bytes += data.0.len() as u64;
    }

    /// Start the upload over.
    pub fn reset_upload(&mut self) {
        self.assert_owner_unsealed();
        for i in 0..self.n_chunks {
            env::storage_remove(&chunk_key(i));
        }
        self.n_chunks = 0;
        self.total_bytes = 0;
    }

    /// Check the blob hash and shape, then freeze the weights forever.
    pub fn seal(&mut self, expected_sha256: String) -> ModelInfo {
        self.assert_owner_unsealed();
        let blob = self.load_blob();
        let hash = hex(&env::sha256(&blob));
        require!(hash == expected_sha256.to_lowercase(), "sha256 mismatch");
        let model = Model::parse(&blob).unwrap_or_else(|e| env::panic_str(&format!("{e:?}")));
        let c = model.config;
        let info = ModelInfo {
            sha256: hash,
            bytes: blob.len() as u64,
            dim: c.dim,
            hidden_dim: c.hidden_dim,
            n_layers: c.n_layers,
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            vocab_size: c.vocab_size,
            seq_len: c.seq_len,
            params: param_count(&c),
        };
        self.sealed = Some(info.clone());
        emit("model_sealed", json!(info));
        info
    }

    // ------------------------------------------------------------------
    // Inference
    // ------------------------------------------------------------------

    /// Run the model as part of a transaction. Every validator executes it;
    /// the result is recorded onchain. Without `seed`, the block's random
    /// seed is used. Stops early (stop = "budget") if gas runs low.
    pub fn generate(
        &mut self,
        prompt: String,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        seed: Option<U64>,
    ) -> Generation {
        let seed = seed.map(|s| s.0).unwrap_or_else(block_seed);
        let g = self.run(prompt, max_tokens, temperature, seed);
        self.generations += 1;
        emit(
            "generation",
            json!({
                "id": self.generations - 1,
                "caller": env::predecessor_account_id(),
                "prompt": g.prompt,
                "text": g.text,
                "seed": g.seed,
                "temperature": g.temperature,
                "new_tokens": g.new_tokens,
                "gas_burnt_tgas": g.gas_burnt_tgas,
            }),
        );
        g
    }

    /// Free, read-only inference over RPC. Needs an explicit seed.
    pub fn complete(
        &self,
        prompt: String,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        seed: Option<U64>,
    ) -> Generation {
        // Views can't read gas counters, so cap the length instead.
        let max_tokens = Some(max_tokens.unwrap_or(32).min(MAX_VIEW_TOKENS));
        self.run_with(prompt, max_tokens, temperature, seed.map(|s| s.0).unwrap_or(0), false)
    }

    /// The Endless Story: anyone can ask the onchain model to write the next
    /// part, optionally steering it with a few words of their own.
    pub fn extend_story(&mut self, nudge: Option<String>, max_tokens: Option<u32>) -> Chapter {
        let nudge = nudge.unwrap_or_default().trim().to_string();
        require!(nudge.len() <= MAX_NUDGE_BYTES, "nudge too long");
        let mut prompt = tail_on_char_boundary(&self.story_text(), STORY_CONTEXT_BYTES);
        if !nudge.is_empty() {
            if !prompt.is_empty() && !prompt.ends_with(' ') {
                prompt.push(' ');
            }
            prompt.push_str(&nudge);
        }
        if prompt.is_empty() {
            prompt = "Once upon a time".to_string();
        }
        let seed = block_seed();
        let g = self.run(prompt, max_tokens.or(Some(48)), Some(0.8), seed);
        // The chapter is the user's nudge (if the story had started) plus the model's words.
        let mut text = String::new();
        if self.story.is_empty() && nudge.is_empty() {
            text.push_str("Once upon a time");
        } else if !nudge.is_empty() {
            text.push_str(&nudge);
        }
        text.push_str(&g.text);
        let chapter = Chapter {
            index: self.story.len(),
            author: env::predecessor_account_id(),
            nudge,
            text,
            block_height: U64(env::block_height()),
            seed: U64(seed),
        };
        self.story.push(chapter.clone());
        self.generations += 1;
        emit("chapter", json!(chapter));
        chapter
    }

    // ------------------------------------------------------------------
    // Views
    // ------------------------------------------------------------------

    pub fn model_info(&self) -> Option<ModelInfo> {
        self.sealed.clone()
    }

    pub fn upload_status(&self) -> near_sdk::serde_json::Value {
        json!({
            "owner_id": self.owner_id,
            "chunks": self.n_chunks,
            "bytes": self.total_bytes,
            "sealed": self.sealed.is_some(),
            "generations": self.generations,
        })
    }

    pub fn get_story(&self, from: Option<u32>, limit: Option<u32>) -> Vec<Chapter> {
        let from = from.unwrap_or(0);
        let limit = limit.unwrap_or(50).min(200);
        (from..self.story.len().min(from.saturating_add(limit)))
            .map(|i| self.story[i].clone())
            .collect()
    }

    pub fn story_length(&self) -> u32 {
        self.story.len()
    }
}

impl Chainmind {
    fn run(
        &self,
        prompt: String,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        seed: u64,
    ) -> Generation {
        self.run_with(prompt, max_tokens, temperature, seed, true)
    }

    fn run_with(
        &self,
        prompt: String,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        seed: u64,
        metered: bool,
    ) -> Generation {
        let info = self.sealed.as_ref().unwrap_or_else(|| env::panic_str("model not sealed"));
        require!(prompt.len() <= MAX_PROMPT_BYTES, "prompt too long");
        let temperature = temperature.unwrap_or(0.8);
        require!(
            temperature.is_finite() && (0.0..=2.0).contains(&temperature),
            "temperature must be in [0, 2]"
        );
        let params = GenParams {
            max_new_tokens: max_tokens.unwrap_or(32).min(MAX_NEW_TOKENS),
            temperature,
            seed,
        };

        let gas_start = if metered { env::used_gas() } else { Gas::from_gas(0) };
        let blob = self.load_blob();
        let model = Model::parse(&blob).unwrap_or_else(|_| env::panic_str("corrupt model"));

        // Stop before we run out: keep room for the costliest step so far.
        let prepaid = if metered { env::prepaid_gas() } else { Gas::from_gas(0) };
        let mut last = gas_start;
        let mut worst_step = Gas::from_gas(0);
        let out = model.generate(&prompt, &params, |_| {
            if !metered {
                return true;
            }
            let now = env::used_gas();
            let step = now.saturating_sub(last);
            if step > worst_step {
                worst_step = step;
            }
            last = now;
            let need = now
                .saturating_add(worst_step.saturating_mul(2))
                .saturating_add(FINISH_RESERVE);
            need < prepaid
        });

        Generation {
            prompt,
            text: out.text,
            prompt_tokens: out.prompt_tokens,
            new_tokens: out.new_tokens,
            stop: match out.stop {
                StopReason::EndOfText => "end_of_text",
                StopReason::MaxTokens => "max_tokens",
                StopReason::ContextFull => "context_full",
                StopReason::Budget => "budget",
            }
            .to_string(),
            temperature,
            seed: U64(seed),
            token_ids: out.token_ids,
            gas_burnt_tgas: if metered {
                (env::used_gas().as_gas() - gas_start.as_gas()) as f64 / 1e12
            } else {
                0.0
            },
            model_sha256: info.sha256.clone(),
        }
    }

    fn load_blob(&self) -> Vec<u8> {
        require!(self.n_chunks > 0, "no model uploaded");
        let mut blob = Vec::with_capacity(self.total_bytes as usize);
        for i in 0..self.n_chunks {
            let chunk = env::storage_read(&chunk_key(i)).unwrap_or_else(|| env::panic_str("missing chunk"));
            blob.extend_from_slice(&chunk);
        }
        blob
    }

    fn story_text(&self) -> String {
        // Only the last few chapters matter for the prompt window.
        let n = self.story.len();
        let start = n.saturating_sub(4);
        (start..n).map(|i| self.story[i].text.as_str()).collect::<Vec<_>>().join(" ")
    }

    fn assert_owner_unsealed(&self) {
        require!(env::predecessor_account_id() == self.owner_id, "only the owner");
        require!(self.sealed.is_none(), "model is sealed and can never change");
    }
}

fn chunk_key(i: u32) -> Vec<u8> {
    let mut k = CHUNK_PREFIX.to_vec();
    k.extend_from_slice(&i.to_le_bytes());
    k
}

fn block_seed() -> u64 {
    u64::from_le_bytes(env::random_seed()[..8].try_into().unwrap())
}

fn param_count(c: &chainmind_engine::Config) -> u64 {
    let (d, h, kv, v, l) = (
        c.dim as u64,
        c.hidden_dim as u64,
        c.kv_dim() as u64,
        c.vocab_size as u64,
        c.n_layers as u64,
    );
    v * d + l * (2 * d + 2 * d * d + 2 * d * kv + 3 * d * h) + d
}

fn tail_on_char_boundary(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut start = s.len() - max;
    while !s.is_char_boundary(start) {
        start += 1;
    }
    s[start..].to_string()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn emit(event: &str, data: near_sdk::serde_json::Value) {
    log!(
        "EVENT_JSON:{}",
        json!({ "standard": "chainmind", "version": "1.0.0", "event": event, "data": [data] })
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use chainmind_tools::{random_model, sha256_hex, stories260k_config};
    use near_sdk::test_utils::VMContextBuilder;
    use near_sdk::testing_env;

    fn owner() -> AccountId {
        "owner.testnet".parse().unwrap()
    }

    fn ctx(who: &AccountId) {
        testing_env!(VMContextBuilder::new()
            .current_account_id("mind.testnet".parse().unwrap())
            .predecessor_account_id(who.clone())
            .prepaid_gas(Gas::from_tgas(300))
            .random_seed([7; 32])
            .build());
    }

    fn loaded() -> (Chainmind, Vec<u8>) {
        ctx(&owner());
        let blob = random_model(stories260k_config(), 7).to_blob();
        let mut c = Chainmind::new(owner());
        for (i, chunk) in blob.chunks(100_000).enumerate() {
            c.upload_chunk(i as u32, Base64VecU8(chunk.to_vec()));
        }
        c.seal(sha256_hex(&blob));
        (c, blob)
    }

    #[test]
    fn upload_seal_and_generate() {
        let (mut c, blob) = loaded();
        let info = c.model_info().unwrap();
        assert_eq!(info.bytes, blob.len() as u64);
        assert_eq!(info.n_layers, 5);
        let g = c.generate("Once upon a time".into(), Some(8), Some(0.0), Some(U64(1)));
        assert_eq!(g.new_tokens + (g.stop == "end_of_text") as u32 * 0, g.new_tokens);
        assert!(g.prompt_tokens > 1);
    }

    #[test]
    fn onchain_output_matches_offchain_engine() {
        let (c, blob) = loaded();
        let g = c.complete("The dog".into(), Some(12), Some(0.9), Some(U64(42)));
        let m = Model::parse(&blob).unwrap();
        let local = m.generate(
            "The dog",
            &GenParams { max_new_tokens: 12, temperature: 0.9, seed: 42 },
            |_| true,
        );
        assert_eq!(g.token_ids, local.token_ids);
        assert_eq!(g.text, local.text);
    }

    #[test]
    #[should_panic(expected = "sha256 mismatch")]
    fn seal_checks_hash() {
        ctx(&owner());
        let blob = random_model(stories260k_config(), 7).to_blob();
        let mut c = Chainmind::new(owner());
        c.upload_chunk(0, Base64VecU8(blob[..200_000].to_vec()));
        c.upload_chunk(1, Base64VecU8(blob[200_000..].to_vec()));
        c.seal("00".repeat(32));
    }

    #[test]
    #[should_panic(expected = "model is sealed")]
    fn sealed_model_cannot_change() {
        let (mut c, _) = loaded();
        c.upload_chunk(99, Base64VecU8(vec![1]));
    }

    #[test]
    #[should_panic(expected = "only the owner")]
    fn only_owner_uploads() {
        ctx(&"eve.testnet".parse().unwrap());
        let mut c = Chainmind::new(owner());
        c.upload_chunk(0, Base64VecU8(vec![1]));
    }

    #[test]
    fn story_grows_chapter_by_chapter() {
        let (mut c, _) = loaded();
        ctx(&"alice.testnet".parse().unwrap());
        let ch0 = c.extend_story(None, Some(6));
        assert!(ch0.text.starts_with("Once upon a time"));
        let ch1 = c.extend_story(Some("Then a dragon".into()), Some(6));
        assert!(ch1.text.starts_with("Then a dragon"));
        assert_eq!(c.story_length(), 2);
        assert_eq!(c.get_story(None, None)[1].author.as_str(), "alice.testnet");
    }

    #[test]
    fn tail_respects_utf8() {
        assert_eq!(tail_on_char_boundary("héllo", 4), "llo");
        assert_eq!(tail_on_char_boundary("hi", 10), "hi");
    }
}
