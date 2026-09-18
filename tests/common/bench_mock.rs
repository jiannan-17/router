//! Minimal worker used by the router overhead harness.
//!
//! Unlike [`super::mock_worker::MockWorker`], it keeps no per-request
//! capture store, parses no JSON and sleeps for nothing, so at tens of
//! thousands of requests per second its own cost stays flat. It answers
//! the endpoints the router touches at startup (`/health`), during health
//! checks and on the benchmarked routes.

use axum::body::Bytes;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

pub struct BenchMockWorker {
    url: String,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl BenchMockWorker {
    /// Bind an ephemeral port on 127.0.0.1 and serve until [`stop`](Self::stop).
    pub async fn start() -> BenchMockWorker {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind bench mock worker");
        let port = listener.local_addr().expect("local addr").port();
        let app = Router::new()
            .route("/health", get(health))
            .route("/health_generate", get(health))
            .route("/v1/models", get(models))
            .route("/v1/completions", post(completion))
            .route("/v1/chat/completions", post(chat_completion))
            .route("/generate", post(generate));
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            });
            if let Err(e) = server.await {
                eprintln!("bench mock worker error: {e}");
            }
        });
        BenchMockWorker {
            url: format!("http://127.0.0.1:{port}"),
            shutdown_tx: Some(shutdown_tx),
            handle: Some(handle),
        }
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

async fn health() -> Response {
    Json(json!({"status": "healthy"})).into_response()
}

async fn models() -> Response {
    Json(json!({
        "object": "list",
        "data": [{"id": "mock-model", "object": "model", "owned_by": "vllm"}]
    }))
    .into_response()
}

// The body is read to completion so the connection can be reused, and then
// dropped without parsing: a worker that does nothing is the floor we are
// measuring the router against.
async fn completion(_body: Bytes) -> Response {
    Json(json!({
        "id": "cmpl-bench",
        "object": "text_completion",
        "created": 0,
        "model": "mock-model",
        "choices": [{"text": "ok", "index": 0, "logprobs": null, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
    .into_response()
}

async fn chat_completion(_body: Bytes) -> Response {
    Json(json!({
        "id": "chatcmpl-bench",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
    .into_response()
}

async fn generate(_body: Bytes) -> Response {
    Json(json!({
        "text": "ok",
        "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}
    }))
    .into_response()
}
