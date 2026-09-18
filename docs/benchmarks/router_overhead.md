# Router CPU fast-path baseline

Two tools measure what the router itself costs on the request path, so
that changes to that path (tokenizer integration, routing-key derivation,
policy work) can be compared against a fixed "before".

| Tool | What it measures | Runs in CI |
|---|---|---|
| `cargo bench --bench routing_input` | Per-call cost of deriving the routing key and selecting a worker, in process, with criterion | compiled only |
| `cargo test --release --test router_overhead_bench -- --ignored --nocapture` | End-to-end router-added latency, throughput, router CPU and peak RSS with mock workers | compiled only |

Both take their inputs from `tests/common/bench_corpus.rs`: seeded,
offline, byte-identical on every machine. `tests/bench_corpus_test.rs`
keeps the corpus and the bench mock worker building and deterministic in CI.

## What these numbers can and cannot say

- They measure router **cost**. The mock workers answer instantly and keep
  no state, so no row here shows a routing **benefit** (KV-cache hits,
  time to first token, load spread). That needs real workers.
- Client-observed latency includes the client, the loopback network and the
  mock worker. The `direct_mock` scenario drives a mock worker with no
  router in the path and is the floor to read the other rows against.
- Means subtract; percentiles do not. The table reports
  `mean - direct_mock mean`, and lists the router and direct percentiles
  side by side. A "router P99" obtained by subtracting two P99 values is
  not a meaningful quantity and is not printed.
- Router CPU and RSS are per process, read with `wait4(2)` for the router
  child only, after it has been stopped. `ru_maxrss` is bytes on macOS and
  kibibytes on Linux; the harness normalizes to bytes.

## Corpus

| Kind | Requests | Models |
|---|---|---|
| `hot64` | 64 fixed prompts, cycled | a cache-hit workload; every request after the first 64 repeats one |
| `cold` | unique marker at the start of every prompt | a cache-miss workload; no two prompts share a prefix |
| `mixed90` | 90% `hot64`, 10% `cold` | a mostly-warm workload |
| `short_shared_prefix` | shared prefix (at most 600 bytes, at most half the prompt) then a unique tail | a shared system prompt with a varying user turn |

Sizes: 200 B, 2 KiB, 16 KiB of ASCII text. Requests are non-streaming
`/v1/completions` (text prompt) or `/v1/chat/completions` (one user
message, no session id) with `max_tokens: 16`.

## Criterion groups (`benches/routing_input.rs`)

- `routing_key/extract_text_for_routing/{completion,chat}/{size}`: the one
  call per request that produces today's routing text.
- `policy/{cache_aware,rendezvous_hash}/{size}`: `select_worker` over four
  healthy workers, tree warmed with the hot set.
- `cache_aware_key_format/{raw_text,one_char_per_token,digit_tagged}/{tokens}`:
  `Tree::insert` and `Tree::prefix_match_with_counts` for three candidate
  key encodings at 128 to 8192 tokens. Input for the discussion of
  token-id routing keys; not a router code path.

```text
cargo bench --bench routing_input
cargo bench --bench routing_input -- --save-baseline main      # on the base commit
cargo bench --bench routing_input -- --baseline main           # on the change
```

## Overhead harness (`tests/router_overhead_bench.rs`)

One `vllm-router` process per scenario (the real binary, built by cargo for
the test), four in-process mock workers with zero delay, a closed-loop
client at concurrency 1 and 64, 5 s warmup and 20 s measurement per cell.

Scenarios:

| Scenario | Router policy | Route |
|---|---|---|
| `direct_mock` | none (client to mock worker) | `/v1/completions` |
| `completions_off` | `cache_aware` (default) | `/v1/completions` |
| `completions_rendezvous` | `rendezvous_hash` | `/v1/completions` |
| `chat_off` | `cache_aware` | `/v1/chat/completions` |

```text
cargo test --release --test router_overhead_bench -- --ignored --nocapture
```

Knobs (environment variables): `VLLM_ROUTER_BENCH_SCENARIOS`,
`VLLM_ROUTER_BENCH_SIZES`, `VLLM_ROUTER_BENCH_CORPORA`,
`VLLM_ROUTER_BENCH_CONCURRENCY`, `VLLM_ROUTER_BENCH_WARMUP_SECS`,
`VLLM_ROUTER_BENCH_MEASURE_SECS`, `VLLM_ROUTER_BENCH_WORKERS`,
`VLLM_ROUTER_BENCH_OUT_DIR`. A quick smoke:

```text
VLLM_ROUTER_BENCH_SCENARIOS=direct_mock,completions_off \
VLLM_ROUTER_BENCH_SIZES=2048 VLLM_ROUTER_BENCH_CONCURRENCY=4 \
VLLM_ROUTER_BENCH_WARMUP_SECS=1 VLLM_ROUTER_BENCH_MEASURE_SECS=3 \
cargo test --release --test router_overhead_bench -- --ignored --nocapture
```

Output: `target/router_overhead/summary.json` (every row, plus commit, OS,
architecture, CPU count and profile), `summary.md` (the table below) and
one `*.router.log` per router run.

To limit the router to two CPUs on Linux, pin the whole harness:
`taskset -c 0,1 cargo test --release ...` (the router inherits the
affinity). `TOKIO_WORKER_THREADS` only sets the router's async worker
count and is not a CPU limit; if used, report it as such.

Both tools are also reachable through `scripts/run_benchmarks.py`
(`--bench routing_input`, `--router-overhead`).

## Reporting

Report with the table the harness prints and this header:

```text
commit: <sha>   rustc: <version>   profile: release
os/arch/cpus: <os> <arch> <n>      pinned: <none | taskset -c ...>
corpus seed: 0x244   warmup/measure: 5 s / 20 s   workers: 4
```

| scenario | size | corpus | c | requests | errors | rps | p50 us | p90 us | p99 us | mean us | mean-direct us | router cpu ms/1k req | router max rss MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

When comparing a change against `main`, run both on the same machine in
the same session, keep `direct_mock` in both runs, and treat differences
inside run-to-run noise (measure it: two runs of `main`) as no change.
