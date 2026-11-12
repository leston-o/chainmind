# Chainmind

**Verifiable AI inside NEAR smart contracts.** Chainmind stores a complete language model in a NEAR contract and runs it during the transaction. Every validator executes the model and gets the exact same answer, so a contract can use AI without trusting an oracle, an API company or special hardware.

The first model is a storyteller: [stories260K](https://huggingface.co/karpathy/tinyllamas) (260,032 parameters), quantized to int8. Once sealed, nobody can change it, switch it off or quietly swap it.

**Live on NEAR testnet:** [`mind-mwdofvuu.leston.testnet`](https://testnet.nearblocks.io/address/mind-mwdofvuu.leston.testnet)

**Live demo:** https://chainmind-theta.vercel.app — the sealed model runs in your browser with the same engine as the contract.

| What | Transaction |
|---|---|
| Model sealed (sha256 `663eb9fb…3227cb48`) | [EQyoHQ61…](https://testnet.nearblocks.io/txns/EQyoHQ613CxHj9LhHavV5VCwtgUe9goupJdYwzpRzEB2) |
| `generate("Once upon a time", seed 1)`, 48 tokens, 194 Tgas | [9BbubPgh…](https://testnet.nearblocks.io/txns/9BbubPghC1nW2yvrmFvuGfbvwytLGE88zAJrw59tTW1K) |
| Endless Story, chapter 0 | [B7dEG1dD…](https://testnet.nearblocks.io/txns/B7dEG1dDM4cd6tjnZczMtBD5dPU19LKow9Y4EAGVmTTr) |

The seed-1 transaction wrote *"Once upon a time, there was a little boy named Tim. Tim loved to discover new friends…"*, the same words the native build and the browser build produce for the same prompt, temperature and seed.

## For judges: check it in 5 minutes

Each step says what to do and what you should see.

1. **Ask the live contract, free.** Install [near-cli-rs](https://github.com/near/near-cli-rs) (`npm install -g near-cli-rs`), then run:
   ```bash
   near contract call-function as-read-only mind-mwdofvuu.leston.testnet complete json-args '{"prompt":"The cat","max_tokens":32,"seed":"7"}' network-config testnet now
   ```
   You should see a story starting *"The cat and a family were playing in the sun."* and `token_ids` starting `[269, 261, 272, 314, 310, 382, …]`, computed inside the NEAR VM.
2. **See a transaction where validators ran the model.** Open [9BbubPgh… on NearBlocks](https://testnet.nearblocks.io/txns/9BbubPghC1nW2yvrmFvuGfbvwytLGE88zAJrw59tTW1K). The receipt log holds a `generation` event with the text *"…there was a little boy named Tim…"* and about 192 Tgas used for inference.
3. **Reproduce that transaction on your machine.** From the repo root:
   ```bash
   cargo run -p chainmind-tools --release --bin chainmind -- run models/stories260k.cmnd "Once upon a time" --max 48 --temp 0.8 --seed 1
   ```
   It prints the same story, word for word.
4. **Run the tests.** `cargo test -p chainmind-engine -p chainmind-tools -p chainmind` passes 18 tests (5 engine, 6 tools, 7 contract). The contract test `onchain_output_matches_offchain_engine` checks that the contract and the plain engine agree token for token.
5. **Open the live demo.** Go to https://chainmind-theta.vercel.app (or run it locally: `python3 -m http.server 8000 --directory site`, then open http://localhost:8000). The hero writes stories live with the same engine and model. Scroll to "Three machines. One story.": all three columns show "32 / 32 match the chain". The third column is computed by your browser on the spot. In "Run the sealed model yourself", prompt "The little robot" with seed 42 and temperature 0.80 gives the fingerprint `563CEC21` on every machine.

## Why it matters

Today a smart contract that wants an AI's opinion has to trust someone to run the model. Chainmind runs it inside consensus:

- **Nobody can fake the answer.** A result is as final as a token transfer.
- **Nobody can change the model.** After `seal`, the weights are fixed forever and pinned by their SHA-256 hash.
- **Nobody can switch it off.** No company, no API key, no server.
- **Anyone can check it.** The same input and seed give the same output on any machine, bit for bit.
- **Other contracts can call it** like a shared library.
- **It's cheap.** A 48-token story used 194 Tgas, about 0.02 NEAR at the base gas price.

What this makes possible: game AI players can prove isn't rigged, NFTs whose text was provably written by the chain, risk models that DeFi contracts consult openly, classifiers that screen AI agents' payments, and contest judges with no hidden human in the loop. Many of these need a small classifier or scorer rather than a storyteller, and those are much cheaper than generating text.

## Try it

You need [near-cli-rs](https://github.com/near/near-cli-rs) (`npm install -g near-cli-rs`).

Free, read-only, no account (up to 48 tokens):

```bash
near contract call-function as-read-only mind-mwdofvuu.leston.testnet complete json-args '{"prompt":"The cat","max_tokens":32,"seed":"7"}' network-config testnet now
```

As a transaction that every validator runs and the chain records (needs a testnet account):

```bash
near contract call-function as-transaction mind-mwdofvuu.leston.testnet generate json-args '{"prompt":"Once upon a time","max_tokens":48}' prepaid-gas '300.0 Tgas' attached-deposit '0 NEAR' sign-as YOUR_ACCOUNT.testnet network-config testnet sign-with-legacy-keychain send
```

Add a chapter to the shared Endless Story, optionally steering it with a few words:

```bash
near contract call-function as-transaction mind-mwdofvuu.leston.testnet extend_story json-args '{"nudge":"Then a dragon"}' prepaid-gas '300.0 Tgas' attached-deposit '0 NEAR' sign-as YOUR_ACCOUNT.testnet network-config testnet sign-with-legacy-keychain send
```

Read the story:

```bash
near contract call-function as-read-only mind-mwdofvuu.leston.testnet get_story json-args '{}' network-config testnet now
```

### Check any onchain answer yourself

Take the prompt, temperature and seed from a transaction and re-run it locally:

```bash
cargo run -p chainmind-tools --release --bin chainmind -- run models/stories260k.cmnd "Once upon a time" --max 48 --temp 0.8 --seed 1
```

It prints the same text and token ids the validators produced.

### Landing page

`site/` is a static page that runs the same engine (compiled to a 72 KB wasm) and the same model in the browser. Serve it with any static server:

```bash
python3 -m http.server 8000 --directory site
```

Then open http://localhost:8000. In the "Three machines. One story." section the browser re-runs one of the NEAR VM test cases and shows "32 / 32 match the chain" when its tokens are identical.

## Determinism: three machines, one story

The engine avoids everything that makes neural networks differ across hardware:

- All matrix products are int8 × int8 → i32 integer dot products (weights and activations are quantized; one f32 scale per row).
- Remaining float maths uses only IEEE-exact operations (`+ - * /`, `sqrt`, `floor`). `exp` is our own polynomial, not libm.
- RoPE sin/cos tables are precomputed by the converter and stored in the model blob, so the chain never needs trig.
- Sampling uses a fully specified xorshift64* generator seeded through splitmix64.
- No SIMD, no threads, no fused multiply-add.

Verified with prompt "The cat", seed 7, temperature 0.8, 32 tokens. All three produce `[269, 261, 272, 314, 310, 382, …]`, identical for all 32 tokens:

| Where | How |
|---|---|
| NEAR VM | The contract wasm run by nearcore's `near-vm-runner` 0.38.1 with protocol 88 mainnet parameters (`vmbench/`) |
| x86-64 native | `chainmind run` (`tools/`) |
| Browser / Node wasm | `web/` build, checked on page load in `site/` |

The contract unit test `onchain_output_matches_offchain_engine` checks the same property as part of `cargo test`.

## Gas

Measured in nearcore's own VM (`vmbench/`), protocol 88, mainnet gas parameters, prompt "Once upon a time":

| Tokens asked | Tokens written | Gas | Stop reason |
|---|---|---|---|
| 1 | 1 | 19.9 Tgas | max_tokens |
| 8 | 8 | 43.4 Tgas | max_tokens |
| 32 | 32 | 130.1 Tgas | max_tokens |
| 64 | 64 | 260.6 Tgas | max_tokens |
| 256 | 69 | 282.7 Tgas | budget |

Loading the 298,843-byte model costs about 16 Tgas; each token about 3.6 Tgas. `generate` watches its own gas and stops cleanly (`stop: "budget"`) before the 300 Tgas limit, returning what it has. On testnet, the 48-token seed-1 call above used 194 Tgas.

One-time costs: uploading the model (2 chunks) 16 + 8 Tgas, sealing 25 Tgas.

## How it works

```
engine/     Deterministic int8 Llama-architecture inference. No dependencies.
contract/   The NEAR contract: weight upload, seal, generate, complete, Endless Story.
tools/      `chainmind` CLI: convert llama2.c checkpoints, make test models, re-run generations.
web/        The engine as a browser wasm with a tiny C ABI (no wasm-bindgen).
site/       Landing page; runs web/ + the model in the browser.
vmbench/    Runs the contract in nearcore's VM to measure gas without a node.
scripts/    Node scripts: deploy to testnet, generate, extend the story.
models/     stories260k.cmnd (the sealed model) and random260k.cmnd (random weights, same shape).
```

### Model format (`CMND` v1)

One little-endian blob holds the model and the tokenizer, so one hash pins both: header and config, int8 embedding table, per layer (RMSNorm weights, q/k/v/o, RMSNorm, SwiGLU w1/w2/w3), final norm, RoPE cos/sin tables, optional classifier, then the SentencePiece-style vocabulary with merge scores. Every int8 matrix stores one f32 scale per row. The converter (`chainmind convert`) reads llama2.c `.bin` checkpoints.

### Contract API

| Method | Kind | What it does |
|---|---|---|
| `new(owner_id)` | init | |
| `upload_chunk(index, data)` | owner, before seal | Appends a base64 chunk of the model blob (max 256 KB). |
| `reset_upload()` | owner, before seal | Starts the upload over. |
| `seal(expected_sha256)` | owner, once | Checks the hash and that the blob parses, then freezes the weights forever. |
| `generate(prompt, max_tokens?, temperature?, seed?)` | call | Runs the model in the transaction. Defaults: 32 tokens, temperature 0.8, seed from the block's randomness. Max 256 tokens; stops early when gas runs low. Emits a `generation` event. |
| `complete(prompt, max_tokens?, temperature?, seed?)` | view | Same, free over RPC. Capped at 48 tokens; seed defaults to 0. |
| `extend_story(nudge?, max_tokens?)` | call | The Endless Story: continues from the last chapters plus your nudge (48 tokens by default) and stores the chapter with author, block and seed. |
| `get_story(from?, limit?)`, `story_length()` | view | Read the story. |
| `model_info()`, `upload_status()` | view | Model shape, hash and parameter count. |

Calling it from another contract:

```rust
#[ext_contract(ext_mind)]
trait Chainmind {
    fn generate(&mut self, prompt: String, max_tokens: Option<u32>,
                temperature: Option<f32>, seed: Option<U64>) -> Generation;
}

ext_mind::ext("mind-mwdofvuu.leston.testnet".parse().unwrap())
    .with_static_gas(Gas::from_tgas(220))
    .generate("The knight opened the chest and found".into(), Some(40), None, None)
    .then(Self::ext(env::current_account_id())
        .with_static_gas(Gas::from_tgas(10))
        .on_text())
```

The whole transaction shares 300 Tgas, so a cross-contract call fits about 50 tokens.

## Build from source

Requirements: Rust 1.95 or newer (near-sdk 5.30's minimum), the `wasm32-unknown-unknown` target, Node 18+ for the scripts.

```bash
rustup target add wasm32-unknown-unknown
cargo test -p chainmind-engine -p chainmind-tools -p chainmind
cargo build -p chainmind --release --target wasm32-unknown-unknown
```

The first `cargo test` takes a while because near-sdk's unit-testing feature compiles parts of nearcore.

Convert the model yourself (download `stories260K.bin` and `tok512.bin` from [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K) into `models/download/`):

```bash
cargo run -p chainmind-tools --release --bin chainmind -- convert models/download/stories260K.bin models/download/tok512.bin models/stories260k.cmnd
```

Measure gas in nearcore's VM (this compiles nearcore and wasmtime, which can take an hour the first time):

```bash
cargo build -p chainmind-vmbench --profile bench-native
./target/bench-native/chainmind-vmbench target/wasm32-unknown-unknown/release/chainmind.wasm models/stories260k.cmnd
```

Rebuild the browser engine:

```bash
cargo build -p chainmind-web --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/chainmind_web.wasm site/engine.wasm
base64 -i models/stories260k.cmnd -o site/stories260k.b64.txt
```

### Deploy your own

Log in or create a testnet account with near-cli-rs so its key is in `~/.near-credentials/testnet/`, then:

```bash
cd scripts && npm install
node deploy.mjs YOUR_ACCOUNT.testnet --fund 8
node generate.mjs "Once upon a time" --as YOUR_ACCOUNT.testnet --max 48 --seed 1
node generate.mjs "The cat" --view --max 32 --seed 7
node generate.mjs --story --as YOUR_ACCOUNT.testnet "Then a dragon"
```

`deploy.mjs` creates a sub-account, deploys the contract, uploads the model, seals it and writes `deployment.json`. About 6 NEAR of the funding is locked as storage for the 308 KB contract and the 299 KB model. Add `--lock` to delete the sub-account's only full-access key after sealing, so nobody, including you, can ever change the code or the model.

The current testnet deployment is **not** locked yet.

## Limits and roadmap

This will never run a chatbot like ChatGPT. It's for small models where trust matters more than size.

- **Per call:** 300 Tgas, about 70 tokens from this model.
- **Bigger models:** work per token grows with model size; a model 10× bigger fits only a handful of tokens per call.
- **Storage:** about 1 NEAR per 100 KB of weights.

Next steps:

1. **One model split across shards.** Put each layer in its own contract and chain the calls; NEAR's sharding lets those contracts share the work.
2. **Checking big models on chain.** Run a large model off chain and post its answer with a deposit. In a dispute, the contract recomputes only the one layer in question. That needs an engine whose maths matches bit for bit off chain and on chain, which is what this engine provides.

For scale: SmolLM2-135M, one of the smallest models Ollama runs, is about 520 times this model. By our estimate it would lock around 1,350 NEAR of storage and need about ten chained calls per token, which is why big models are better checked on chain than run there.

## NEAR Hackathon 2026

Built for the NEAR Hackathon (1–14 Nov 2026). Track: **Open Innovation** (`open`).

All code in this repository was written during the hackathon window, starting 12 Nov 2026; there is no pre-hackathon baseline.

## Built with

Libraries, tools and outside work used in this project (hackathon rule Q-005):

- **Rust:** [near-sdk](https://crates.io/crates/near-sdk) 5.30 for the contract; [near-vm-runner](https://crates.io/crates/near-vm-runner), near-parameters and near-primitives-core 0.38.1 (nearcore) for the gas benchmark; sha2 and base64.
- **JavaScript:** [near-api-js](https://github.com/near/near-api-js) 5 for the deploy and generate scripts; [GSAP](https://gsap.com) 3.12.5 with ScrollTrigger for the landing page animations; fonts Bricolage Grotesque, Geist and Geist Mono from Google Fonts.
- **Tools:** near-cli-rs, Google Chrome (headless, for recording the demo), ffmpeg, Python with numpy and scipy (the demo soundtrack is synthesized by `demo/music.py`).
- **Model and design:** stories260K weights and tokenizer from llama2.c; the inference loop follows llama2.c's `run.c`.

## Credits and licence

Chainmind is MIT licensed (see `LICENSE`). The stories260K model and tokenizer are by Andrej Karpathy from [llama2.c](https://github.com/karpathy/llama2.c) (MIT), trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. The inference design follows llama2.c's `run.c`, rewritten in Rust with integer matmuls for determinism.
