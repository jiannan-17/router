//! Router overhead harness: what the router adds on top of a worker that
//! does nothing, measured end to end on one machine.
//!
//! One `vllm-router` process (the real binary, release profile recommended)
//! is spawned per scenario in front of in-process mock workers that answer
//! instantly. A closed-loop client drives the router, and the harness
//! reports client-observed latency, throughput and the router process's own
//! CPU time and peak RSS, taken per process with `wait4(2)` after the
//! process is stopped. The `direct_mock` scenario drives a mock worker
//! without a router and is the floor every other row is read against.
//!
//! Numbers from this harness describe router cost only. Nothing here can
//! show a routing *benefit* (KV-cache hits, time to first token); that needs
//! real workers.
//!
//! Ignored by default. Run with:
//!
//! ```text
//! cargo test --release --test router_overhead_bench -- --ignored --nocapture
//! ```
//!
//! Knobs (environment variables, all optional):
//! - `VLLM_ROUTER_BENCH_SCENARIOS`: comma list of `direct_mock`,
//!   `completions_off`, `completions_rendezvous`, `chat_off` (default: all)
//! - `VLLM_ROUTER_BENCH_SIZES`: prompt bytes, default `200,2048,16384`
//! - `VLLM_ROUTER_BENCH_CORPORA`: `hot64,cold,mixed90,short_shared_prefix`,
//!   default `hot64,cold`
//! - `VLLM_ROUTER_BENCH_CONCURRENCY`: default `1,64`
//! - `VLLM_ROUTER_BENCH_WARMUP_SECS` / `VLLM_ROUTER_BENCH_MEASURE_SECS`:
//!   default `5` / `20`
//! - `VLLM_ROUTER_BENCH_WORKERS`: mock workers behind the router, default `4`
//! - `VLLM_ROUTER_BENCH_OUT_DIR`: default `target/router_overhead`
//!
//! See `docs/benchmarks/router_overhead.md` for the method and the report
//! template.

#![cfg(unix)]

mod common;

use common::bench_corpus::{chat, completion_text, Corpus, CorpusKind, SIZES};
use common::bench_mock::BenchMockWorker;
use reqwest::header::CONTENT_TYPE;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Completions,
    Chat,
}

impl Route {
    fn path(self) -> &'static str {
        match self {
            Route::Completions => "/v1/completions",
            Route::Chat => "/v1/chat/completions",
        }
    }

    fn body(self, prompt: &str) -> String {
        match self {
            Route::Completions => completion_text(prompt).to_string(),
            Route::Chat => chat(None, prompt).to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    name: &'static str,
    /// `None` drives a mock worker directly (no router).
    policy: Option<&'static str>,
    route: Route,
}

const SCENARIOS: [Scenario; 4] = [
    Scenario {
        name: "direct_mock",
        policy: None,
        route: Route::Completions,
    },
    Scenario {
        name: "completions_off",
        policy: Some("cache_aware"),
        route: Route::Completions,
    },
    Scenario {
        name: "completions_rendezvous",
        policy: Some("rendezvous_hash"),
        route: Route::Completions,
    },
    Scenario {
        name: "chat_off",
        policy: Some("cache_aware"),
        route: Route::Chat,
    },
];

struct Config {
    scenarios: Vec<Scenario>,
    sizes: Vec<usize>,
    corpora: Vec<CorpusKind>,
    concurrency: Vec<usize>,
    warmup: Duration,
    measure: Duration,
    workers: usize,
    out_dir: PathBuf,
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

impl Config {
    fn from_env() -> Config {
        let scenarios = env_list("VLLM_ROUTER_BENCH_SCENARIOS", "")
            .into_iter()
            .map(|n| {
                *SCENARIOS
                    .iter()
                    .find(|s| s.name == n)
                    .unwrap_or_else(|| panic!("unknown scenario {n}"))
            })
            .collect::<Vec<_>>();
        let scenarios = if scenarios.is_empty() {
            SCENARIOS.to_vec()
        } else {
            scenarios
        };
        let default_sizes = SIZES
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        Config {
            scenarios,
            sizes: env_list("VLLM_ROUTER_BENCH_SIZES", &default_sizes)
                .iter()
                .map(|s| s.parse().expect("size"))
                .collect(),
            corpora: env_list("VLLM_ROUTER_BENCH_CORPORA", "hot64,cold")
                .iter()
                .map(|c| CorpusKind::parse(c).unwrap_or_else(|| panic!("unknown corpus {c}")))
                .collect(),
            concurrency: env_list("VLLM_ROUTER_BENCH_CONCURRENCY", "1,64")
                .iter()
                .map(|c| c.parse().expect("concurrency"))
                .collect(),
            warmup: Duration::from_secs(env_u64("VLLM_ROUTER_BENCH_WARMUP_SECS", 5)),
            measure: Duration::from_secs(env_u64("VLLM_ROUTER_BENCH_MEASURE_SECS", 20)),
            workers: env_u64("VLLM_ROUTER_BENCH_WORKERS", 4) as usize,
            out_dir: PathBuf::from(
                std::env::var("VLLM_ROUTER_BENCH_OUT_DIR")
                    .unwrap_or_else(|_| "target/router_overhead".to_string()),
            ),
        }
    }
}

/// Resource usage of one child process, read with `wait4(2)` after it exited.
#[derive(Clone, Copy, Debug, Default, Serialize)]
struct ProcessUsage {
    user_cpu_s: f64,
    sys_cpu_s: f64,
    max_rss_bytes: u64,
}

struct RouterProcess {
    child: Child,
    port: u16,
}

impl RouterProcess {
    fn spawn(worker_urls: &[String], policy: &str, log_path: &PathBuf) -> RouterProcess {
        let port = portpicker::pick_unused_port().expect("router port");
        let prometheus_port = portpicker::pick_unused_port().expect("prometheus port");
        let log = fs::File::create(log_path).expect("router log file");
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_vllm-router"));
        cmd.arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--prometheus-port")
            .arg(prometheus_port.to_string())
            .arg("--policy")
            .arg(policy)
            .arg("--log-level")
            .arg("warn")
            .arg("--worker-startup-check-interval")
            .arg("1")
            .arg("--worker-urls");
        for url in worker_urls {
            cmd.arg(url);
        }
        let child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .expect("spawn vllm-router");
        RouterProcess { child, port }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    async fn wait_ready(&mut self, client: &reqwest::Client) {
        let deadline = Instant::now() + Duration::from_secs(60);
        let health = format!("{}/health", self.url());
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                panic!("vllm-router exited before becoming ready: {status}");
            }
            if let Ok(resp) = client.get(&health).send().await {
                if resp.status().is_success() {
                    return;
                }
            }
            assert!(
                Instant::now() < deadline,
                "vllm-router did not become ready"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Stop the router and return its own resource usage. `SIGKILL` is used
    /// on purpose: the load has already stopped, and CPU time and peak RSS
    /// are accounted regardless of how the process ends.
    fn stop(self) -> ProcessUsage {
        let pid = self.child.id() as libc::pid_t;
        // SAFETY: plain libc calls on a pid we spawned and still own; `rusage`
        // is a POD struct that wait4 fills in.
        unsafe {
            libc::kill(pid, libc::SIGKILL);
            let mut status: libc::c_int = 0;
            let mut usage: libc::rusage = std::mem::zeroed();
            let waited = libc::wait4(pid, &mut status, 0, &mut usage);
            assert_eq!(waited, pid, "wait4 failed for the router process");
            ProcessUsage {
                user_cpu_s: timeval_secs(usage.ru_utime),
                sys_cpu_s: timeval_secs(usage.ru_stime),
                max_rss_bytes: max_rss_bytes(usage.ru_maxrss),
            }
        }
    }
}

fn timeval_secs(tv: libc::timeval) -> f64 {
    tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6
}

/// `ru_maxrss` is bytes on macOS and kibibytes on Linux and the BSDs.
fn max_rss_bytes(ru_maxrss: libc::c_long) -> u64 {
    if cfg!(target_os = "macos") {
        ru_maxrss as u64
    } else {
        ru_maxrss as u64 * 1024
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct LoadResult {
    requests: u64,
    errors: u64,
    measure_secs: f64,
    latencies_us: Vec<u64>,
}

/// Closed-loop load: `concurrency` tasks, each sending the next request as
/// soon as the previous response body has been read. Request `i` of the run
/// uses `corpus.prompt(i)`; tasks interleave indices so every corpus kind
/// sees the sequence it was designed for.
async fn run_load(
    client: &reqwest::Client,
    base_url: &str,
    route: Route,
    corpus: Arc<Corpus>,
    concurrency: usize,
    warmup: Duration,
    measure: Duration,
) -> LoadResult {
    let url = format!("{base_url}{}", route.path());
    let start = Instant::now();
    let warm_end = start + warmup;
    let end = warm_end + measure;
    // Pre-serialized bodies for the fixed hot set keep client-side JSON work
    // out of the loop; the other corpora build each body on the fly.
    let hot_bodies: Arc<Vec<String>> = Arc::new(match corpus.kind() {
        CorpusKind::Hot64 => (0..64).map(|i| route.body(&corpus.prompt(i))).collect(),
        _ => Vec::new(),
    });

    let mut tasks = Vec::with_capacity(concurrency);
    for task in 0..concurrency {
        let client = client.clone();
        let url = url.clone();
        let corpus = Arc::clone(&corpus);
        let hot_bodies = Arc::clone(&hot_bodies);
        tasks.push(tokio::spawn(async move {
            let mut latencies = Vec::with_capacity(8192);
            let mut errors = 0u64;
            let mut i = task;
            loop {
                let now = Instant::now();
                if now >= end {
                    break;
                }
                let body = if hot_bodies.is_empty() {
                    route.body(&corpus.prompt(i))
                } else {
                    hot_bodies[i % hot_bodies.len()].clone()
                };
                let t0 = Instant::now();
                let ok = match client
                    .post(&url)
                    .header(CONTENT_TYPE, "application/json")
                    .body(body)
                    .send()
                    .await
                {
                    Ok(resp) => {
                        let ok = resp.status().is_success();
                        let _ = resp.bytes().await;
                        ok
                    }
                    Err(_) => false,
                };
                if t0 >= warm_end {
                    if ok {
                        latencies.push(t0.elapsed().as_micros() as u64);
                    } else {
                        errors += 1;
                    }
                }
                i += concurrency;
            }
            (latencies, errors)
        }));
    }

    let mut result = LoadResult {
        measure_secs: measure.as_secs_f64(),
        ..Default::default()
    };
    for task in tasks {
        let (latencies, errors) = task.await.expect("load task");
        result.requests += latencies.len() as u64;
        result.errors += errors;
        result.latencies_us.extend(latencies);
    }
    result.latencies_us.sort_unstable();
    result
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

#[derive(Clone, Debug, Serialize)]
struct Row {
    scenario: String,
    policy: Option<String>,
    route: String,
    size_bytes: usize,
    corpus: String,
    concurrency: usize,
    workers: usize,
    warmup_secs: f64,
    measure_secs: f64,
    requests: u64,
    errors: u64,
    rps: f64,
    p50_us: u64,
    p90_us: u64,
    p99_us: u64,
    mean_us: f64,
    /// `mean_us` minus the `direct_mock` mean for the same size/corpus/
    /// concurrency, when that row was measured in this run. Means subtract;
    /// percentiles do not, so no such column exists for them.
    mean_minus_direct_us: Option<f64>,
    router: Option<ProcessUsage>,
    router_cpu_ms_per_1k_requests: Option<f64>,
}

fn row_key(r: &Row) -> (usize, String, usize) {
    (r.size_bytes, r.corpus.clone(), r.concurrency)
}

fn git_head() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

#[derive(Serialize)]
struct Report {
    commit: String,
    os: &'static str,
    arch: &'static str,
    cpus: usize,
    profile: &'static str,
    rows: Vec<Row>,
}

fn render_markdown(rows: &[Row]) -> String {
    let mut out = String::new();
    out.push_str("| scenario | size | corpus | c | requests | errors | rps | p50 us | p90 us | p99 us | mean us | mean-direct us | router cpu ms/1k req | router max rss MiB |\n");
    out.push_str("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for r in rows {
        let cpu = r
            .router_cpu_ms_per_1k_requests
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let rss = r
            .router
            .map(|u| format!("{:.1}", u.max_rss_bytes as f64 / (1024.0 * 1024.0)))
            .unwrap_or_else(|| "-".to_string());
        let delta = r
            .mean_minus_direct_us
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "-".to_string());
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.0} | {} | {} | {} | {:.1} | {} | {} | {} |\n",
            r.scenario,
            r.size_bytes,
            r.corpus,
            r.concurrency,
            r.requests,
            r.errors,
            r.rps,
            r.p50_us,
            r.p90_us,
            r.p99_us,
            r.mean_us,
            delta,
            cpu,
            rss
        ));
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "load test: run explicitly with --ignored --nocapture"]
async fn router_overhead() {
    let cfg = Config::from_env();
    fs::create_dir_all(&cfg.out_dir).expect("out dir");
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(256)
        .timeout(Duration::from_secs(30))
        .build()
        .expect("client");

    let mut workers = Vec::with_capacity(cfg.workers);
    for _ in 0..cfg.workers {
        workers.push(BenchMockWorker::start().await);
    }
    let worker_urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    eprintln!(
        "router overhead harness: commit={} os={} arch={} cpus={} profile={} workers={} warmup={:?} measure={:?}",
        git_head(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        profile,
        cfg.workers,
        cfg.warmup,
        cfg.measure
    );
    if profile == "debug" {
        eprintln!("warning: debug profile; use `cargo test --release ...` for reportable numbers");
    }

    let mut rows: Vec<Row> = Vec::new();
    for scenario in &cfg.scenarios {
        for &size in &cfg.sizes {
            for &kind in &cfg.corpora {
                for &concurrency in &cfg.concurrency {
                    let corpus = Arc::new(Corpus::new(kind, size));
                    let label = format!(
                        "{}-{}-{}-c{}",
                        scenario.name,
                        size,
                        kind.name(),
                        concurrency
                    );
                    eprintln!("== {label}");
                    let (load, usage) = match scenario.policy {
                        None => (
                            run_load(
                                &client,
                                &worker_urls[0],
                                scenario.route,
                                corpus,
                                concurrency,
                                cfg.warmup,
                                cfg.measure,
                            )
                            .await,
                            None,
                        ),
                        Some(policy) => {
                            let log_path = cfg.out_dir.join(format!("{label}.router.log"));
                            let mut router = RouterProcess::spawn(&worker_urls, policy, &log_path);
                            router.wait_ready(&client).await;
                            let load = run_load(
                                &client,
                                &router.url(),
                                scenario.route,
                                corpus,
                                concurrency,
                                cfg.warmup,
                                cfg.measure,
                            )
                            .await;
                            (load, Some(router.stop()))
                        }
                    };
                    let mean_us = if load.latencies_us.is_empty() {
                        0.0
                    } else {
                        load.latencies_us.iter().sum::<u64>() as f64
                            / load.latencies_us.len() as f64
                    };
                    let row = Row {
                        scenario: scenario.name.to_string(),
                        policy: scenario.policy.map(str::to_string),
                        route: scenario.route.path().to_string(),
                        size_bytes: size,
                        corpus: kind.name().to_string(),
                        concurrency,
                        workers: cfg.workers,
                        warmup_secs: cfg.warmup.as_secs_f64(),
                        measure_secs: load.measure_secs,
                        requests: load.requests,
                        errors: load.errors,
                        rps: load.requests as f64 / load.measure_secs,
                        p50_us: percentile(&load.latencies_us, 0.50),
                        p90_us: percentile(&load.latencies_us, 0.90),
                        p99_us: percentile(&load.latencies_us, 0.99),
                        mean_us,
                        mean_minus_direct_us: None,
                        router: usage,
                        router_cpu_ms_per_1k_requests: usage.map(|u| {
                            if load.requests == 0 {
                                0.0
                            } else {
                                (u.user_cpu_s + u.sys_cpu_s) * 1000.0
                                    / (load.requests as f64 / 1000.0)
                            }
                        }),
                    };
                    eprintln!(
                        "   requests={} errors={} rps={:.0} p50={}us p99={}us mean={:.1}us",
                        row.requests, row.errors, row.rps, row.p50_us, row.p99_us, row.mean_us
                    );
                    rows.push(row);
                }
            }
        }
    }

    for w in &mut workers {
        w.stop().await;
    }

    // Fill in the mean delta against direct_mock for matching rows.
    let direct: HashMap<(usize, String, usize), f64> = rows
        .iter()
        .filter(|r| r.policy.is_none())
        .map(|r| (row_key(r), r.mean_us))
        .collect();
    for r in rows.iter_mut() {
        if r.policy.is_some() {
            r.mean_minus_direct_us = direct.get(&row_key(r)).map(|d| r.mean_us - d);
        }
    }

    let report = Report {
        commit: git_head(),
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        profile,
        rows: rows.clone(),
    };
    let json_path = cfg.out_dir.join("summary.json");
    fs::write(
        &json_path,
        serde_json::to_string_pretty(&report).expect("json"),
    )
    .expect("write");
    let md = render_markdown(&rows);
    let md_path = cfg.out_dir.join("summary.md");
    fs::write(&md_path, &md).expect("write");
    eprintln!(
        "\n{md}\nwritten: {} and {}",
        json_path.display(),
        md_path.display()
    );

    let total_errors: u64 = rows.iter().map(|r| r.errors).sum();
    assert_eq!(total_errors, 0, "requests failed during the benchmark");
}
