//! Compile-and-run guard for the benchmark corpus and the bench mock worker.
//!
//! The criterion benchmark and the `#[ignore]` overhead harness are not run
//! in CI; this test keeps their shared inputs building and deterministic.

mod common;

use common::bench_corpus::{
    chat, chat_with_session, completion_ids, completion_text, to_chat_request,
    to_completion_request, Corpus, CorpusKind, HOT_SET_SIZE, SHORT_SHARED_PREFIX_MAX_BYTES, SIZES,
};
use common::bench_mock::BenchMockWorker;
use vllm_router_rs::protocols::spec::GenerationRequest;
use vllm_router_rs::tokenizer::huggingface::HuggingFaceTokenizer;
use vllm_router_rs::tokenizer::traits::Encoder;

const FIXTURE: &str = "tests/fixtures/tokenizer/byte_level_bpe.json";

#[test]
fn every_corpus_builds_prompts_of_the_requested_size() {
    for kind in CorpusKind::ALL {
        for size in SIZES {
            let corpus = Corpus::new(kind, size);
            for i in [0usize, 1, 9, 10, 63, 64, 65, 1000] {
                let prompt = corpus.prompt(i);
                assert_eq!(prompt.len(), size, "{} size={} i={}", kind.name(), size, i);
                assert!(prompt.is_ascii());
            }
        }
    }
}

#[test]
fn corpus_is_deterministic_across_instances() {
    for kind in CorpusKind::ALL {
        let a = Corpus::new(kind, 2048);
        let b = Corpus::new(kind, 2048);
        for i in 0..200 {
            assert_eq!(a.prompt(i), b.prompt(i), "{} i={}", kind.name(), i);
        }
    }
}

#[test]
fn hot64_cycles_and_cold_never_repeats() {
    let hot = Corpus::new(CorpusKind::Hot64, 2048);
    assert_eq!(hot.prompt(3), hot.prompt(3 + HOT_SET_SIZE));
    assert_ne!(hot.prompt(3), hot.prompt(4));

    let cold = Corpus::new(CorpusKind::Cold, 2048);
    let mut seen = std::collections::HashSet::new();
    for i in 0..500 {
        assert!(seen.insert(cold.prompt(i)), "cold prompt {} repeated", i);
    }
    // The unique marker sits at the start, so no two cold prompts share a prefix.
    let first_word = |s: String| s.split(' ').nth(1).map(str::to_string);
    assert_ne!(first_word(cold.prompt(1)), first_word(cold.prompt(2)));
}

#[test]
fn mixed90_is_ten_percent_cold() {
    let corpus = Corpus::new(CorpusKind::Mixed90, 2048);
    let hot = Corpus::new(CorpusKind::Hot64, 2048);
    let cold = (0..100)
        .filter(|&i| corpus.prompt(i) != hot.prompt(i))
        .count();
    assert_eq!(cold, 10);
}

#[test]
fn short_shared_prefix_shares_a_bounded_prefix() {
    for size in SIZES {
        let corpus = Corpus::new(CorpusKind::ShortSharedPrefix, size);
        let prefix = corpus.shared_prefix();
        assert!(!prefix.is_empty());
        assert!(prefix.len() <= SHORT_SHARED_PREFIX_MAX_BYTES);
        assert!(prefix.len() <= size / 2);
        let a = corpus.prompt(1);
        let b = corpus.prompt(2);
        assert!(a.starts_with(prefix) && b.starts_with(prefix));
        assert_ne!(a, b);
    }
}

#[test]
fn builders_produce_typed_requests_with_expected_routing_text() {
    let corpus = Corpus::new(CorpusKind::Hot64, 200);
    let prompt = corpus.prompt(0);

    let completion = to_completion_request(&completion_text(&prompt));
    assert_eq!(completion.extract_text_for_routing(), prompt);

    let tokenizer = HuggingFaceTokenizer::from_file(FIXTURE).expect("fixture tokenizer");
    let ids: Vec<u32> = tokenizer
        .encode(&prompt)
        .expect("encode")
        .token_ids()
        .to_vec();
    assert!(!ids.is_empty());
    let by_ids = to_completion_request(&completion_ids(&ids));
    assert_eq!(
        by_ids.extract_text_for_routing(),
        format!("token_ids:{}", ids.len())
    );

    let chat_req = to_chat_request(&chat(Some("system"), &prompt));
    assert_eq!(chat_req.extract_text_for_routing(), "");
    let session = to_chat_request(&chat_with_session("s-1", &prompt));
    assert_eq!(session.extract_text_for_routing(), "s-1");
}

#[tokio::test]
async fn bench_mock_worker_answers_health_and_generation_routes() {
    let mut worker = BenchMockWorker::start().await;
    let client = reqwest::Client::new();
    let health = client
        .get(format!("{}/health", worker.url()))
        .send()
        .await
        .expect("health");
    assert!(health.status().is_success());

    let body = completion_text("hello");
    let resp = client
        .post(format!("{}/v1/completions", worker.url()))
        .json(&body)
        .send()
        .await
        .expect("completion");
    assert!(resp.status().is_success());
    let json: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(json["object"], "text_completion");

    let resp = client
        .post(format!("{}/v1/chat/completions", worker.url()))
        .json(&chat(None, "hello"))
        .send()
        .await
        .expect("chat");
    assert!(resp.status().is_success());
    worker.stop().await;
}
