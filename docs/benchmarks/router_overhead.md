# Router CPU fast-path baseline

Two tools measure what the router itself costs on the request path, so
that changes to that path (tokenizer integration, routing-key derivation,
policy work) can be compared against a fixed "before".

| Tool | What it measures | In CI |
|---|---|---|
| `cargo bench --bench routing_input` | Per-call cost of deriving the routing key and selecting a worker, in process, with criterion | built by the clippy job (`--all-targets`); never run |
| `cargo test --release --test router_overhead_bench -- --ignored --nocapture` | End-to-end router-added latency, throughput, load spread across workers, router CPU and peak RSS with mock workers | built by the integration-test and clippy jobs; ignored, never run |

Both take their inputs from `tests/common/bench_corpus.rs`: seeded,
offline, byte-identical on every machine. `tests/bench_corpus_test.rs`
keeps the corpus and the bench mock worker building and deterministic in CI.
Router edge cases (long prompts, threshold and load-balance boundaries,
non-ASCII keys, stale tenants, session headers, the body limit) live in
`tests/common/routing_edge.rs`; `tests/routing_edge_cases_test.rs` asserts
them in CI and the `edge/*` criterion groups time the same fixtures. See
[Edge cases](#edge-cases).

## What these numbers can and cannot say

- They measure router **cost**. The mock workers answer instantly and keep
  no state, so no row here shows a routing **benefit** (KV-cache hits,
  time to first token). That needs real workers with prefix caching on.
- Client-observed latency includes the client, the loopback network and the
  mock worker. The `direct_completions` and `direct_chat` scenarios drive a
  mock worker with no router in the path and are the floor to read the other
  rows against. Each routed row is read against the direct row for its own
  route: a chat request and its response are not the same work as a
  completion one, so the two floors differ.
- Means subtract; percentiles do not. The table reports
  `mean - direct mean` on the same route, and lists the router and direct
  percentiles side by side. A "router P99" obtained by subtracting two P99
  values is not a meaningful quantity and is not printed.
- Router CPU and RSS are per process, read with `wait4(2)` for the router
  child only, after it has been stopped. `ru_maxrss` is bytes on macOS and
  kibibytes on Linux; the harness normalizes to bytes. `router cpu ms/1k
  req` is the router's CPU over its whole life divided by every request it
  served in the cell (warmup and measurement); startup is a small fixed
  cost included in it, noticeable only in low-throughput cells.
- The cache-aware tree is never evicted inside a cell: every router gets
  `--eviction-interval 3600` and starts empty. For corpora that do not
  repeat (`cold`, `long_shared_prefix`, `short_shared_prefix`), each
  `cache_aware` request adds to the tree, so its CPU and RSS in those rows
  include tree growth, which grows with requests times prompt size. That is
  what the policy does between eviction passes in production, stated here
  so it is not mistaken for per-request parsing cost.
- The client and the mock workers run on one 8-thread runtime inside the
  harness; the router is a separate process with its default thread count.
  On a machine without that many spare cores, router rows include CPU
  contention the direct rows do not. The report records both thread counts
  and the harness's own CPU per cell.
- `hotspot` is the largest worker's share of the generation requests
  divided by the even share `1 / workers`: `1.0` is a perfectly even
  spread, `workers` means every request landed on one worker. It shows
  whether a routing policy concentrates load; it says nothing about
  whether that concentration was useful.
- Run-to-run noise is real. With `VLLM_ROUTER_BENCH_REPEATS` above one the
  summary reports the median per cell and the spread of the mean
  (`(max - min) / median`); differences inside that spread are not
  findings.

## Comparing routing designs: four arms

When evaluating a token-aware routing change, run four arms on the same
request set, at the same offered load and output length, with the same
model and tokenizer version:

| Arm | Routing key | Router-side encoding |
|---|---|---|
| A | the current policy | none |
| B | text-prefix hash | none |
| C | token-prefix hash | every request, cache off |
| D | token-prefix hash | cache on |

B against A isolates prefix routing itself, C against B the cost of token
boundaries, D against C the cache, and only D against A and B decides
whether the whole change is worth it. D faster than C but slower than A or
B only shows that the cache offsets a cost the change introduced.

Today the harness ships arm A (`completions_off`, `completions_rendezvous`,
`chat_off`) and the floors (`direct_completions`, `direct_chat`); arms B to
D are added by the change under evaluation.

## Corpus

| Kind | Requests | Question it answers |
|---|---|---|
| `hot64` | 64 fixed prompts, cycled | the upper bound of an exact-match cache; every request after the first 64 repeats one |
| `long_shared_prefix` | all but the last 64 bytes shared, unique tail | same prefix, varying suffix: an exact-match cache misses every time while a prefix router sees one prefix |
| `short_shared_prefix` | shared prefix (at most 600 bytes, at most half the prompt) then a unique tail | many requests sharing one system prompt: does the policy create a hotspot |
| `cold` | unique marker at the start of every prompt | no repeats and no shared prefix: the pure added cost |
| `mixed90` | 90% `hot64`, 10% `cold` | a mostly-warm workload |
| `utf8_hot64` | `hot64` with CJK and emoji text after the ASCII marker | the cache-aware tree's non-ASCII paths on repeated long keys |

Text that is not meant to be shared comes from a pool of 256 distinct
seeded bodies (fewer above 16 KiB: at most 64 MiB per pool, at least 8
bodies). Routing-key code is content dependent (substring search,
hashing, parsing), and one repeated body lets the CPU's branch predictors
learn it: with a single body, `rendezvous_hash` selection on 16 KiB prompts
measured about three times cheaper than on distinct prompts.

Sizes: 200 B, 2 KiB, 16 KiB of ASCII text by default; 128 KiB, 512 KiB and
1 MiB for the long-input runs. Sizes are prompt bytes. `utf8_hot64` has
the same byte sizes and fewer characters. Requests are non-streaming
`/v1/completions` (text prompt) or `/v1/chat/completions` (one user
message, no session id) with `max_tokens: 16`.

Pre-tokenized prompts (`IdCorpus`, reported as `token_ids`) are sized by
id count, 16384 and 131072 by default, not by bytes. Ids are drawn from
10000 to 31999, so every id is five digits and a request body is exactly
`envelope + 6 × ids - 1` bytes. Every report row carries `body_bytes`, the
serialized request size, next to the input size.

## Criterion groups (`benches/routing_input.rs`)

- `routing_key/extract_text_for_routing/{completion,chat}/{size}`: the one
  call per request that produces today's routing text.
- `policy/{cache_aware,rendezvous_hash}/{size}`: `select_worker` over four
  healthy workers, tree warmed with the hot set.
- `cache_aware_key_format/{raw_text,one_char_per_token,digit_tagged}/{tokens}`:
  `Tree::insert` and `Tree::prefix_match_with_counts` for three candidate
  key encodings at 128 to 8192 tokens. Input for the discussion of
  token-id routing keys; not a router code path.
- `edge/*`: the router edge cases, described in [Edge cases](#edge-cases).
  `cargo bench --bench routing_input -- edge/` runs only these (about six
  minutes).

```text
cargo bench --bench routing_input
cargo bench --bench routing_input -- --save-baseline main      # on the base commit
cargo bench --bench routing_input -- --baseline main           # on the change
```

## Overhead harness (`tests/router_overhead_bench.rs`)

One `vllm-router` process per scenario cell (the real binary, built by
cargo for the test), four in-process mock workers with zero delay, a
client at concurrency 1 and 64, 5 s warmup and 20 s measurement per cell.
The client is closed loop by default; `VLLM_ROUTER_BENCH_RATE` caps the
offered load so arms can be compared at equal load (each task then paces
itself to `rate / concurrency`, still with one request in flight).

Scenarios:

| Scenario | Router policy | Route |
|---|---|---|
| `direct_completions` | none (client to mock worker) | `/v1/completions` |
| `direct_chat` | none (client to mock worker) | `/v1/chat/completions` |
| `completions_off` | `cache_aware` (default) | `/v1/completions` |
| `completions_rendezvous` | `rendezvous_hash` | `/v1/completions` |
| `chat_off` | `cache_aware` | `/v1/chat/completions` |

```text
cargo test --release --test router_overhead_bench -- --ignored --nocapture
```

Knobs (environment variables): `VLLM_ROUTER_BENCH_SCENARIOS`,
`VLLM_ROUTER_BENCH_SIZES`, `VLLM_ROUTER_BENCH_CORPORA`,
`VLLM_ROUTER_BENCH_PROMPT_IDS`,
`VLLM_ROUTER_BENCH_CONCURRENCY`, `VLLM_ROUTER_BENCH_RATE`,
`VLLM_ROUTER_BENCH_REPEATS`, `VLLM_ROUTER_BENCH_WARMUP_SECS`,
`VLLM_ROUTER_BENCH_MEASURE_SECS`, `VLLM_ROUTER_BENCH_WORKERS`,
`VLLM_ROUTER_BENCH_ROUTER_CPUS`, `VLLM_ROUTER_BENCH_OUT_DIR`. A quick
smoke:

```text
VLLM_ROUTER_BENCH_SCENARIOS=direct_completions,completions_off \
VLLM_ROUTER_BENCH_SIZES=2048 VLLM_ROUTER_BENCH_CONCURRENCY=4 \
VLLM_ROUTER_BENCH_WARMUP_SECS=1 VLLM_ROUTER_BENCH_MEASURE_SECS=3 \
cargo test --release --test router_overhead_bench -- --ignored --nocapture
```

Each row is keyed by its input: `text<bytes>` for text prompts or
`ids<count>` for pre-tokenized ones, never a bare number. Id cells run on
the completions scenarios only and are not crossed with sizes and corpora.
Every row also reports `harness_cores`, the CPU the client and the mock
workers used divided by the wall-clock time of the same interval (router
start to router stop). It is a hint, not a bottleneck test: the client can
limit throughput without keeping every runtime thread busy, especially at
low concurrency.

Output: `target/router_overhead/summary.json` (every run, plus commit and
whether tracked files were modified, rustc version, OS, architecture, CPU
count, profile, thread counts, router pinning, tree settings, repeats and
rate), `summary.md` (a per-run table and a median-over-runs table) and one
`*.router.log` per router run. With repeats, the scenario order is rotated
every repeat. Request bodies are built the same way for every corpus,
outside the timed region.

To limit only the router to some CPUs on Linux, set
`VLLM_ROUTER_BENCH_ROUTER_CPUS=0,1`: the harness runs the router under
`taskset -c 0,1` and leaves the client and the mock workers unpinned.
Pinning the whole `cargo test` process instead would put the client on the
same cores and measure a client-bound system. `TOKIO_WORKER_THREADS` only
sets the router's async worker count and is not a CPU limit; the report
records it as `router_worker_threads`.

Both tools are also reachable through `scripts/run_benchmarks.py`
(`--bench routing_input`, `--router-overhead`).

## Edge cases

`tests/common/routing_edge.rs` defines each case once. The test asserts it
in CI; the benchmark asserts the same branch once and then times it.

### Fixture configuration

Every cache-aware case sets its configuration in full:
`cache_threshold 0.5`, `balance_abs_threshold 5`,
`balance_rel_threshold 2.0`, eviction off. Three configurations exist and
they differ:

| Source | threshold | abs | rel | eviction |
|---|---:|---:|---:|---|
| edge-case fixture | 0.5 | 5 | 2.0 | off |
| `CacheAwareConfig::default()` | 0.5 | 32 | 1.1 | 30 s |
| `vllm-router` CLI defaults (the overhead harness) | 0.3 | 64 | 1.5 | 120 s (the harness passes 3600 s) |

The small balance thresholds let single-digit loads cross them. The 45%
and 55% cases sit on either side of the fixture's 0.5; with the CLI
default of 0.3 both would be hits.

Four workers with loads `[1, 1, 1, 0]` and the key warmed onto W1 make the
selected worker name the branch: **W1** is a cache hit, **W3** (the only
least-loaded worker) is a low-match miss or the imbalanced path, and
**W0** (the first healthy worker while W1 is down) is the stale-tenant
fallback.

`select_worker` inserts every prompt it routes, so the benchmark restores
the starting state before every timed call: it removes every worker from
the tree (which prunes it to the root), warms the keys again and resets
the loads, all outside the timed region. Criterion plans iteration counts
from wall time, setup included, so this shortens the run's iteration
count rather than its duration. Long inputs are built once per case and
borrowed; no case uses `iter_batched`, which would hold many long inputs at
once.

### Cases

| Case | Code path | Benchmark | Assertion |
|---|---|---|---|
| long prompt, full hit (128 KiB, 512 KiB, 1 MiB) | walk and re-insert the whole key | `edge/cache_aware/long_hit/{size}` | W1 |
| long prompt, cold | miss; the whole prompt becomes a new leaf | `edge/cache_aware/long_cold/{size}` | W3 |
| long request body | the handler's `axum::Json` extraction into the typed request, `extract_text_for_routing` (a full copy), `serde_json::to_vec` to forward it | `edge/long_input/{deserialize,extract,serialize}/{completion,chat}/{size}`, and `deserialize/{completion,chat}_utf8/{size}` for CJK text | exact sizes |
| 45% / 55% of the probe shared | `match_rate > cache_threshold` | `edge/cache_aware/threshold_{45,55}pct/{16KiB,1MiB}` | W3 / W1 |
| exactly half shared, and one character either side (2000 B and 1 MiB) | the comparison is `>`, not `>=` | none | W3 / W3 / W1, and `prefix_match_with_counts` matches exactly the shared characters |
| imbalanced load `[1,6,1,0]` | skips the prefix lookup, still inserts the prompt for the least-loaded worker | `edge/cache_aware/imbalanced/16KiB` | W3 |
| only `abs` exceeded `[11,16,11,10]`, only `rel` `[2,6,2,1]`, `abs` equal `[1,5,1,0]`, `rel` equal `[7,12,7,6]` | both conditions of `max - min > abs && max > min × rel` | none | W1 (balanced) |
| 1 MiB CJK key: full hit, 45% of the characters shared | non-ASCII prefix counting and slicing (`shared_prefix_count_chars`, `advance_by_chars`, `chars().count()`) | `edge/cache_aware/utf8_{hit,45pct}/1MiB` | W1 / W3 |
| character ratio 0.40 with byte ratio 0.67; character ratio 0.60 with byte ratio 0.30 | the match rate counts characters | none | W3 / W1 |
| fork inside a multi-byte character (`中` and `丰` share two bytes), exactly half | the byte scan stops mid-character and falls back to counting characters | none | W3 |
| hit on a tenant that is down | fallback to the first healthy worker, stale tenant removed | none | W0; after W1 recovers, W3 |
| long prompts with `x-session-id` (`rendezvous_hash`) | the header short-circuits the key scan and hash | `edge/rendezvous/{body,header}/{size}` | two prompts that differ without the header land together with it; an empty header value is ignored |
| `"user": "u-1"` inside the prompt text | `rendezvous_hash` scans the routing text for JSON-like fields | `edge/rendezvous/json_like_prompt/{16KiB,1MiB}` | input only |
| 16384 and 131072 token ids | `IntArray` parsing; the routing text is `token_ids:<count>` | `edge/token_ids/{deserialize,extract}/ids{n}` | input only |
| 1, 4, 16, 64 workers | per-worker hashing; tenants per tree node | `edge/workers/{rendezvous,cache_aware}/{n}` | every hot prompt hits its own tenant |
| body of exactly `--max-payload-size` and one byte more (1 MiB) | `DefaultBodyLimit` and `RequestBodyLimitLayer` | none | 200, then 413 |

Two readings to keep in mind:

- The `header` rendezvous rows cost the same at every size because the
  header skips the policy's text scans and hash. The request path does not
  get cheaper to the same degree: `route_typed_request` copies the full
  prompt with `extract_text_for_routing` before it calls the policy, and
  the body is still parsed and serialized. Read those rows together with
  the `edge/long_input` rows.
- Three behaviors on `main` are recorded here, not changed:
  - A text prompt on `/v1/completions` is parsed through the untagged
    `PromptInput`, which tries its string variant last. Each of the three
    variants that fail first formats the whole prompt into its error
    message, so a long completion request costs several times a chat
    request of the same size to parse, and far more for non-ASCII text
    (compare the `deserialize/*_utf8` rows).
  - A prompt that contains `"user": "…"` is keyed by that field under
    `rendezvous_hash`.
  - Every pre-tokenized prompt of the same length has the same routing text
    (`token_ids:<count>`). With a fixed set of healthy workers,
    `rendezvous_hash` sends all of them to one worker. `cache_aware` keeps
    them on one worker too, until the load becomes imbalanced and it
    switches to the least-loaded worker.

### Long inputs end to end

The harness refuses prompts of 128 KiB or more on `cold`, `mixed90` and
`short_shared_prefix`: every request adds about its size to the
cache-aware tree, which is never evicted inside a cell. Name the corpora
explicitly:

```bash
VLLM_ROUTER_BENCH_SIZES=131072,524288,1048576 \
VLLM_ROUTER_BENCH_CORPORA=hot64,utf8_hot64,long_shared_prefix \
VLLM_ROUTER_BENCH_PROMPT_IDS=131072 \
VLLM_ROUTER_BENCH_CONCURRENCY=1,16 \
cargo test --release --test router_overhead_bench -- --ignored --nocapture
```

Request bodies are built before the timed region but still cost the
client CPU, so long-input throughput may be client-bound. `harness_cores`
hints at that; confirming it takes a control run with the bodies built in
advance.

## Reporting

Report the median-over-runs table the harness prints and this header:

```text
commit: <sha> (dirty: <bool>)   rustc: <version>   profile: release
os/arch/cpus: <os> <arch> <n>   router cpus: <none | taskset list>   router threads: <default | n>
corpus seed: 0x244   warmup/measure: <w> s / <m> s   workers: 4   repeats: <n>   rate: <closed loop | n rps>
```

When comparing a change against `main`, run both on the same machine in
the same session with nothing else running, keep the direct scenarios in
both runs, use at least three repeats, and treat differences inside the
reported spread as no change.
