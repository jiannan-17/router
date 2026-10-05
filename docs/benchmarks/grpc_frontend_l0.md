# gRPC frontend L0 benchmark

This is the gRPC frontend companion to #290's offline overhead harness. It
runs on main without depending on that unmerged PR. A local byte-level BPE
fixture renders real chat requests through `EngineFrontend::prepare` and
sends the prepared IDs to the same mock gRPC worker used by integration tests.
No GPU, model download, or inference engine is needed.

```bash
cargo test --release --test grpc_frontend_bench -- --ignored --nocapture
```

The harness prints one JSON object per cell. Settings follow #290's
`VLLM_ROUTER_BENCH_` convention:

| Suffix | Default | Meaning |
|---|---|---|
| `SIZES` | `200,16384,131072` | User-message bytes; minimum 32 |
| `CONCURRENCY` | `1,8` | Closed-loop clients |
| `SECONDS` | `2` | Measurement duration per cell |

Each cell has a fresh frontend and compares L0 off/on. `hot64` cycles through
64 warmed prompts; `cold` uses unique markers disjoint from warmup. All
prompts have the requested byte length. The 64 warmup requests load the
model, establish the connection, and fill the hot set before timing.
Every measured response must succeed. The harness asserts that hot requests
hit and cold requests miss with L0 enabled. Defaults use one 64 MiB budget,
10,000 entries, and a 1 MiB per-entry limit.

Reported fields include encode P50/P99, prepare-through-response-body P50/P99,
requests/s, process CPU seconds and CPU microseconds/request, cache bytes,
entries, hits, and misses. Cache occupancy includes the warmup set; hit/miss
counts exclude warmup. CPU includes the driver, request construction, frontend,
and mock worker in the same process. Latency excludes request construction,
but includes rendering, encoding, protobuf transport, response adaptation,
and body consumption. This measures frontend/transport overhead, not the
external HTTP listener, routing policies, model execution, or KV-cache reuse.
The Tokio runtime has eight worker threads. All outputs are non-streaming;
streaming parity is covered by `grpc_frontend_l0` integration tests.

For less noisy measurements, increase `SECONDS` and repeat on an idle host.
The small fixture's vocabulary and merge rules do not represent every real
model tokenizer; compare production assets separately before choosing budgets.

## Local results

Apple M4 Pro, macOS 26.6.2, Rust 1.98.1, release profile; 2026-10-05.
Source: main `1034096` plus this change. One pass, 2 seconds per cell,
eight Tokio worker threads, 64 warmup requests per cell. All 24 cells
completed without request errors. Numbers below are **L0 off → on**;
retained bytes are for L0 on (off retains zero). CPU covers the entire
in-process harness as described above. These short runs are indicative,
not statistical confidence intervals.

| Input | Corpus | Clients | Encode P50 µs | Latency P50 µs | Latency P99 µs | Requests/s | CPU µs/request | Cache MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 200 B | hot64 | 1 | 44.4 → 0.2 | 118.0 → 60.0 | 193.1 → 85.4 | 8,012.2 → 16,062.0 | 394.0 → 75.5 | 0.07 |
| 200 B | cold | 1 | 44.4 → 46.6 | 118.2 → 119.5 | 193.7 → 193.0 | 8,024.3 → 7,977.3 | 387.2 → 362.1 | 10.20 |
| 200 B | hot64 | 8 | 56.0 → 0.5 | 340.1 → 231.4 | 522.4 → 354.0 | 23,241.1 → 33,651.8 | 355.5 → 154.2 | 0.07 |
| 200 B | cold | 8 | 55.3 → 56.4 | 337.1 → 340.0 | 521.0 → 526.6 | 23,432.8 → 23,243.1 | 353.8 → 356.5 | 10.20 |
| 16 KiB | hot64 | 1 | 620.0 → 5.6 | 821.3 → 151.5 | 953.6 → 199.8 | 1,206.2 → 6,398.1 | 1,188.7 → 170.3 | 4.30 |
| 16 KiB | cold | 1 | 590.9 → 622.8 | 793.0 → 830.2 | 946.6 → 952.2 | 1,248.1 → 1,198.1 | 1,257.2 → 1,267.8 | 63.99 |
| 16 KiB | hot64 | 8 | 1,348.9 → 5.9 | 3,071.0 → 375.6 | 5,548.3 → 468.5 | 2,537.0 → 20,942.0 | 1,990.4 → 218.9 | 4.30 |
| 16 KiB | cold | 8 | 1,416.8 → 1,235.8 | 3,138.7 → 2,886.0 | 5,538.0 → 5,063.3 | 2,502.3 → 2,712.5 | 2,024.9 → 1,855.1 | 63.99 |
| 128 KiB | hot64 | 1 | 4,319.2 → 30.9 | 5,093.9 → 746.4 | 5,473.7 → 865.6 | 194.8 → 1,309.2 | 5,987.2 → 793.9 | 34.32 |
| 128 KiB | cold | 1 | 4,294.0 → 4,418.4 | 5,065.9 → 5,181.6 | 5,590.5 → 5,400.0 | 195.5 → 192.1 | 5,941.2 → 6,032.1 | 63.82 |
| 128 KiB | hot64 | 8 | 9,597.2 → 35.2 | 21,323.8 → 2,337.8 | 34,414.7 → 2,773.9 | 376.8 → 3,379.3 | 11,786.8 → 830.1 | 34.32 |
| 128 KiB | cold | 8 | 7,541.6 → 6,767.6 | 17,463.5 → 17,147.9 | 31,888.0 → 29,650.1 | 446.3 → 461.6 | 10,132.5 → 9,194.6 | 63.82 |

All measured `hot64` requests hit and all `cold` requests miss. Cold
workloads have no reusable encodings, so small off/on differences should
be treated as measurement noise or cache bookkeeping cost, not a cache
speedup. Long cold workloads reached the shared byte budget and evicted
entries. The 128 KiB hot set retained 34.32 MiB across 64 entries.

