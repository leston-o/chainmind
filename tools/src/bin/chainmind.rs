//! chainmind convert <model.bin> <tokenizer.bin> <out.cmnd>
//! chainmind random <out.cmnd>
//! chainmind info <model.cmnd>
//! chainmind run <model.cmnd> "<prompt>" [--max N] [--temp T] [--seed S]
//!
//! `run` is the off-chain verifier: give it the same prompt, temperature and
//! seed as an onchain call and it prints the same text and token ids.

use chainmind_engine::{GenParams, Model};
use chainmind_tools::{random_model, read_llama2c, sha256_hex, stories260k_config};
use std::process::exit;
use std::time::Instant;

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    exit(1)
}

fn read(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| die(format!("{path}: {e}")))
}

fn flag<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    match args.iter().position(|a| a == name) {
        Some(i) => args
            .get(i + 1)
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| die(format!("bad value for {name}"))),
        None => default,
    }
}

fn write_blob(path: &str, blob: &[u8]) {
    std::fs::write(path, blob).unwrap_or_else(|e| die(format!("{path}: {e}")));
    println!("wrote {path}: {} bytes, sha256 {}", blob.len(), sha256_hex(blob));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("convert") if args.len() == 4 => {
            let m = read_llama2c(&read(&args[1]), &read(&args[2])).unwrap_or_else(|e| die(e));
            println!("{:?}", m.config);
            write_blob(&args[3], &m.to_blob());
        }
        Some("random") if args.len() == 2 => {
            write_blob(&args[1], &random_model(stories260k_config(), 7).to_blob());
        }
        Some("info") if args.len() == 2 => {
            let blob = read(&args[1]);
            let m = Model::parse(&blob).unwrap_or_else(|e| die(format!("{e:?}")));
            println!("{:?}", m.config);
            println!("{} bytes, sha256 {}", blob.len(), sha256_hex(&blob));
        }
        Some("run") if args.len() >= 3 => {
            let blob = read(&args[1]);
            let m = Model::parse(&blob).unwrap_or_else(|e| die(format!("{e:?}")));
            let p = GenParams {
                max_new_tokens: flag(&args, "--max", 64),
                temperature: flag(&args, "--temp", 0.0),
                seed: flag(&args, "--seed", 0),
            };
            let t = Instant::now();
            let out = m.generate(&args[2], &p, |_| true);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            println!("{}{}", args[2], out.text);
            eprintln!(
                "\n[{} prompt + {} new tokens, stop={:?}, {:.1} ms]",
                out.prompt_tokens, out.new_tokens, out.stop, ms
            );
            eprintln!("token_ids {:?}", out.token_ids);
        }
        _ => {
            eprintln!(
                "usage:\n  chainmind convert <model.bin> <tokenizer.bin> <out.cmnd>\n  \
                 chainmind random <out.cmnd>\n  chainmind info <model.cmnd>\n  \
                 chainmind run <model.cmnd> \"<prompt>\" [--max N] [--temp T] [--seed S]"
            );
            exit(2);
        }
    }
}
