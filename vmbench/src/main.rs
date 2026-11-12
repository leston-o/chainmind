//! Measure Chainmind gas in nearcore's own VM (validation, finite-wasm gas
//! instrumentation and host costs as on mainnet), without a node.
//!
//!   cargo run -p chainmind-vmbench --release -- <chainmind.wasm> <model.cmnd>

use base64::Engine as _;
use near_parameters::RuntimeConfigStore;
use near_primitives_core::types::{Balance, Gas};
use near_primitives_core::version::PROTOCOL_VERSION;
use near_vm_runner::logic::mocks::mock_external::MockedExternal;
use near_vm_runner::logic::{ReturnData, VMContext};
use near_vm_runner::ContractCode;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::Arc;

struct Chain {
    ext: MockedExternal,
    config: Arc<near_parameters::vm::Config>,
    fees: Arc<near_parameters::RuntimeFeesConfig>,
    height: u64,
}

struct Outcome {
    tgas: f64,
    ret: Option<Value>,
    error: Option<String>,
}

impl Chain {
    fn call(&mut self, method: &str, args: Value, view: bool) -> Outcome {
        self.height += 1;
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&self.height.to_le_bytes());
        let ctx = VMContext {
            current_account_id: "mind.near".parse().unwrap(),
            signer_account_id: "mind.near".parse().unwrap(),
            signer_account_pk: vec![0; 33],
            predecessor_account_id: "mind.near".parse().unwrap(),
            refund_to_account_id: "mind.near".parse().unwrap(),
            input: std::rc::Rc::from(serde_json::to_vec(&args).unwrap()),
            promise_results: Vec::new().into(),
            block_height: self.height,
            block_timestamp: 1_763_000_000_000_000_000 + self.height * 1_000_000_000,
            epoch_height: 1,
            account_balance: Balance::from_near(1000),
            account_locked_balance: Balance::ZERO,
            storage_usage: 1_000_000,
            account_contract: near_primitives_core::account::AccountContract::None,
            attached_deposit: Balance::ZERO,
            prepaid_gas: Gas::from_teragas(300),
            random_seed: seed.to_vec(),
            view_config: view.then(|| near_primitives_core::config::ViewConfig {
                max_gas_burnt: Gas::from_teragas(300),
            }),
            output_data_receivers: vec![],
        };
        let gas_counter = ctx.make_gas_counter(&self.config);
        let prepared = near_vm_runner::prepare(&self.ext, self.config.clone(), None, gas_counter, method);
        let out = near_vm_runner::run(prepared, &mut self.ext, &ctx, self.fees.clone())
            .unwrap_or_else(|e| panic!("vm runner error: {e:?}"));
        let ret = match &out.return_data {
            ReturnData::Value(v) if !v.is_empty() => serde_json::from_slice(v).ok(),
            _ => None,
        };
        Outcome {
            tgas: out.burnt_gas.as_gas() as f64 / 1e12,
            ret,
            error: out.aborted.as_ref().map(|e| format!("{e:?}")),
        }
    }

    fn must(&mut self, method: &str, args: Value) -> Outcome {
        let o = self.call(method, args, false);
        if let Some(e) = &o.error {
            panic!("{method} failed: {e}");
        }
        o
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let wasm = std::fs::read(&args[0]).expect("wasm");
    let blob = std::fs::read(&args[1]).expect("model");
    let sha: String = Sha256::digest(&blob).iter().map(|b| format!("{b:02x}")).collect();

    let store = RuntimeConfigStore::new(None);
    let rc = store.get_config(PROTOCOL_VERSION);
    let code = ContractCode::new(wasm, None);
    let mut ext = MockedExternal::with_code(code);
    ext.code_hash = *ext.code.as_ref().unwrap().hash();
    let mut chain = Chain {
        ext,
        config: rc.wasm_config.clone(),
        fees: rc.fees.clone(),
        height: 100,
    };
    println!(
        "protocol {PROTOCOL_VERSION}, vm {:?}, max gas/receipt 300 Tgas\nmodel {} ({} bytes)\n",
        chain.config.vm_kind,
        &args[1],
        blob.len()
    );

    let o = chain.must("new", json!({ "owner_id": "mind.near" }));
    println!("new: {:.1} Tgas", o.tgas);
    const CHUNK: usize = 200 * 1024;
    for (i, c) in blob.chunks(CHUNK).enumerate() {
        let data = base64::engine::general_purpose::STANDARD.encode(c);
        let o = chain.must("upload_chunk", json!({ "index": i, "data": data }));
        println!("upload_chunk {i}: {:.1} Tgas", o.tgas);
    }
    let o = chain.must("seal", json!({ "expected_sha256": sha }));
    println!("seal: {:.1} Tgas -> {}\n", o.tgas, o.ret.unwrap());

    println!("{:<22} {:>6} {:>6} {:>13} {:>9} {:>10}", "call", "asked", "got", "stop", "tx Tgas", "inference");
    for (prompt, max) in [
        ("Once upon a time", 1u32),
        ("Once upon a time", 8),
        ("Once upon a time", 32),
        ("Once upon a time", 64),
        ("Once upon a time", 256),
    ] {
        let o = chain.call(
            "generate",
            json!({ "prompt": prompt, "max_tokens": max, "temperature": 0.8, "seed": "42" }),
            false,
        );
        match (o.ret, o.error) {
            (Some(g), None) => {
                println!(
                    "{:<22} {:>6} {:>6} {:>13} {:>9.1} {:>10.1}",
                    "generate",
                    max,
                    g["new_tokens"],
                    g["stop"].as_str().unwrap(),
                    o.tgas,
                    g["gas_burnt_tgas"].as_f64().unwrap()
                );
                println!("    {}", json!(format!("{prompt}{}", g["text"].as_str().unwrap())));
            }
            (_, e) => println!("generate max={max}: FAILED after {:.1} Tgas: {e:?}", o.tgas),
        }
    }

    let o = chain.call(
        "complete",
        json!({ "prompt": "The cat", "max_tokens": 32, "temperature": 0.8, "seed": "7" }),
        true,
    );
    let g = o.ret.unwrap_or_else(|| panic!("view failed after {:.1} Tgas: {:?}", o.tgas, o.error));
    println!(
        "\nview complete(): {} tokens, {:.1} Tgas, stop={}\n    ids {}",
        g["new_tokens"], o.tgas, g["stop"], g["token_ids"]
    );

    for nudge in [None, Some("Then a dragon")] {
        let o = chain.must("extend_story", json!({ "nudge": nudge }));
        println!("extend_story: {:.1} Tgas -> {}", o.tgas, o.ret.unwrap()["text"]);
    }
}
