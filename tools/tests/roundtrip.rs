use chainmind_engine::{GenParams, Model, StopReason};
use chainmind_tools::{random_model, stories260k_config};

fn blob() -> Vec<u8> {
    random_model(stories260k_config(), 7).to_blob()
}

#[test]
fn blob_parses_with_expected_shape() {
    let b = blob();
    let m = Model::parse(&b).expect("parse");
    assert_eq!(m.config, stories260k_config());
    assert_eq!(m.tokenizer.vocab_size(), 512);
    // ~260K int8 weights + scales + RoPE tables + tokenizer.
    assert!(b.len() > 250_000 && b.len() < 350_000, "size {}", b.len());
}

#[test]
fn truncated_blob_is_rejected() {
    let b = blob();
    assert!(Model::parse(&b[..b.len() - 1]).is_err());
    let mut extra = b.clone();
    extra.push(0);
    assert!(Model::parse(&extra).is_err());
}

#[test]
fn generation_is_deterministic() {
    let b = blob();
    let m = Model::parse(&b).unwrap();
    let p = GenParams {
        max_new_tokens: 20,
        temperature: 0.8,
        seed: 1234,
    };
    let a = m.generate("Once upon a time", &p, |_| true);
    let c = m.generate("Once upon a time", &p, |_| true);
    assert_eq!(a.token_ids, c.token_ids);
    assert_eq!(a.text, c.text);
    assert!(a.prompt_tokens > 1);

    let other = m.generate("Once upon a time", &GenParams { seed: 99, ..p }, |_| true);
    assert_ne!(a.token_ids, other.token_ids);
}

#[test]
fn budget_callback_stops_generation() {
    let b = blob();
    let m = Model::parse(&b).unwrap();
    let p = GenParams {
        max_new_tokens: 50,
        temperature: 0.0,
        seed: 0,
    };
    let out = m.generate("Hi", &p, |pos| pos < 5);
    assert_eq!(out.stop, StopReason::Budget);
}

#[test]
fn max_new_tokens_is_exact() {
    let b = blob();
    let m = Model::parse(&b).unwrap();
    for n in [0, 1, 7] {
        let p = GenParams {
            max_new_tokens: n,
            temperature: 0.0,
            seed: 0,
        };
        let out = m.generate("Once upon a time", &p, |_| true);
        if out.stop == StopReason::MaxTokens {
            assert_eq!(out.new_tokens, n);
        } else {
            assert!(out.new_tokens <= n);
        }
    }
}

#[test]
fn tokenizer_roundtrips_text_and_bytes() {
    let b = blob();
    let m = Model::parse(&b).unwrap();
    for text in ["ab cd", "Once upon a time", "naïve ✓\n"] {
        let ids = m.tokenizer.encode(text, true);
        assert_eq!(ids[0], 1);
        let mut out = Vec::new();
        for w in ids.windows(2) {
            out.extend(m.tokenizer.decode(w[0], w[1]));
        }
        // The dummy prefix space is stripped after BOS.
        assert_eq!(String::from_utf8(out).unwrap(), text, "{ids:?}");
    }
}
