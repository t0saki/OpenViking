//! Rust side of the microbenchmarks; same fixtures and steps as bench/pymicro.py.
//! Usage: microbench FIXTURES_DIR [ITERS]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ov_gateway_rs::anthropic::{encode, Anthropic, SseDecoder};
use ov_gateway_rs::canon::{digest, prefix_chain};
use ov_gateway_rs::kernel::{self, Credential, Ctx};
use ov_gateway_rs::ov::Viking;
use ov_gateway_rs::store::Store;
use ov_gateway_rs::text::{self, Headers, ToolNames};
use serde_json::{json, Value};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn cpu_now() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 * 1000.0 + ts.tv_nsec as f64 / 1e6
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn measure<T>(name: &str, fixture: &str, iters: usize, mut f: impl FnMut() -> T) {
    std::hint::black_box(f());
    let (mut cpu, mut wall) = (vec![], vec![]);
    for _ in 0..5 {
        let (c0, w0) = (cpu_now(), Instant::now());
        for _ in 0..iters {
            std::hint::black_box(f());
        }
        cpu.push((cpu_now() - c0) / iters as f64);
        wall.push(w0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
    }
    println!(
        "{}",
        json!({"name": name, "fixture": fixture, "cpu_ms": (median(cpu) * 10000.0).round() / 10000.0,
        "wall_ms": (median(wall) * 10000.0).round() / 10000.0})
    );
}

fn headers() -> Headers {
    vec![
        ("content-type".into(), "application/json".into()),
        ("anthropic-version".into(), "2023-06-01".into()),
        ("x-claude-code-session-id".into(), "fixture-session".into()),
    ]
}

fn relay(chunks: &[Vec<u8>]) -> Vec<bytes::Bytes> {
    let mut adapter = Anthropic::new();
    adapter.begin();
    let mut decoder = SseDecoder::default();
    let mut out = Vec::new();
    let mut frames = Vec::new();
    let mut received = 0;
    for chunk in chunks {
        received += chunk.len();
        decoder.feed(chunk, &mut frames);
        for f in frames.drain(..) {
            let Some(v) = SseDecoder::data(&f) else {
                continue;
            };
            if v.get("error").is_some() || v["type"] == "error" {
                panic!("error event");
            }
            for e in adapter.event(v).unwrap() {
                out.push(encode(&e));
            }
        }
    }
    std::hint::black_box(received);
    let round = adapter.end().unwrap();
    for e in adapter.publish_calls(&round.calls) {
        out.push(encode(&e));
    }
    let final_message = adapter.final_message(&round.usage);
    // capture.nonstream(final): the assembled reply, as bookkeeping reads it.
    let output = vec![json!({"role": "assistant", "content": final_message["content"].clone()})];
    std::hint::black_box(output);
    for e in adapter.terminal(&final_message) {
        out.push(encode(&e));
    }
    out
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("FIXTURES_DIR"));
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let catalog: Value =
        serde_json::from_slice(&std::fs::read(dir.join("tools.json")).unwrap()).unwrap();
    let policy = kernel::default_policy();
    let tools = kernel::select_tools(&catalog, &policy);
    let names: Vec<String> = tools
        .iter()
        .map(|t| {
            t["function"]["name"]
                .as_str()
                .unwrap()
                .trim_start_matches("openviking_")
                .to_string()
        })
        .collect();
    let tool_names = ToolNames::new(&names);
    let hdrs = headers();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();

    for label in ["44k", "140k"] {
        let raw = std::fs::read(dir.join(format!("body_{label}.json"))).unwrap();
        let body: Value = serde_json::from_slice(&raw).unwrap();
        let messages = body["messages"].as_array().unwrap();
        let chain = prefix_chain(messages.iter());
        println!(
            "{}",
            json!({"anchor": label, "last": chain.last(), "count": chain.iter().filter(|a| !a.is_empty()).count()})
        );
        measure("parse_body", label, iters, || {
            serde_json::from_slice::<Value>(&raw).unwrap()
        });
        measure("prefix_chain", label, iters, || {
            prefix_chain(messages.iter())
        });
        if std::env::var("MICRO_DETAIL").is_ok() {
            let mut buf = Vec::new();
            measure("detail_normalize_only", label, iters, || {
                buf.clear();
                for m in messages {
                    ov_gateway_rs::canon::write_norm(&mut buf, m);
                }
                buf.len()
            });
            let data = buf.clone();
            measure("detail_sha256_only", label, iters, || {
                use sha2::Digest;
                sha2::Sha256::digest(&data)
            });
        }
        measure("classify", label, iters, || {
            text::classify(&body, &hdrs, messages)
        });
        measure("plugin_present", label, iters, || {
            text::plugin_present(&body, &hdrs)
        });
        measure("client_openviking_tools", label, iters, || {
            text::client_openviking_tools(&body, &tool_names)
        });
        measure("thinking_digests", label, iters, || {
            messages
                .iter()
                .filter(|m| m["role"] == "assistant")
                .filter_map(|m| m["content"].as_array())
                .flatten()
                .filter(|b| b["type"] == "text")
                .filter_map(|b| b["text"].as_str())
                .filter(|t| !t.is_empty())
                .map(digest)
                .collect::<std::collections::HashSet<_>>()
        });
        measure("estimate_full_body", label, iters, || text::estimate(&body));
        measure("capture_turns_full", label, iters, || {
            kernel::capture_turns(messages, &chain, 0, messages.len(), true, 0.0)
        });
        measure("dumps_body", label, iters, || {
            ov_gateway_rs::canon::to_vec(&body)
        });

        // Full preparation of a steady-state tool-result request on an existing session.
        let tmp = std::env::temp_dir().join(format!("ovgw-micro-{}-{label}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = Arc::new(Store::open(&tmp));
        store.save_object("bench", "policies", "default", &policy);
        let http = reqwest::Client::builder().no_proxy().build().unwrap();
        let cred = Credential {
            id: "cred".into(),
            account: "bench".into(),
            user_id: "alice".into(),
            openviking_key: "k".into(),
            policy_id: "default".into(),
            upstream_id: "mock-anthropic".into(),
        };
        let ctx = Arc::new(Ctx {
            store: store.clone(),
            viking: Viking::new(http.clone(), "http://127.0.0.1:1"),
            http,
            upstream_base: "http://127.0.0.1:1".into(),
            upstream_key: "x".into(),
            credentials: HashMap::new(),
        });
        let scope = digest("bench\0alice\0anthropic");
        let sid = digest("fixture-session");
        store.replay_put(
            &scope,
            &sid,
            "tools",
            "",
            &json!({"tools": tools, "tool_skip_reason": ""}),
        );
        let full = || {
            runtime.block_on(async {
                let policy = {
                    let s = ctx.store.clone();
                    kernel::blocking(move || s.get_object("bench", "policies", "default"))
                        .await
                        .unwrap()
                };
                let parsed: Value = serde_json::from_slice(&raw).unwrap();
                kernel::prepare(&ctx, parsed, &hdrs, &cred, policy)
                    .await
                    .unwrap()
            })
        };
        let first = full();
        assert!(
            first.tools_active,
            "gateway tools must be active, as in Python"
        );
        let usage = json!({"usage": {"input_tokens": 100000, "output_tokens": 500, "anchor": chain[chain.len() - 2], "time": 1.0}});
        assert!(store.state_swap(&scope, &sid, 0, &usage));
        measure("full_prepare", label, (iters / 2).max(5), full);
        let prepared = full();
        let output = vec![
            json!({"role": "assistant", "content": [{"type": "text", "text": "x ".repeat(3000)}]}),
        ];
        measure("post_reply_chain", label, iters, || {
            prefix_chain(prepared.messages().iter().chain(output.iter()))
        });
        let _ = std::fs::remove_dir_all(&tmp);
    }

    for n in [200, 800, 1500] {
        let chunks: Vec<String> =
            serde_json::from_slice(&std::fs::read(dir.join(format!("sse_{n}.json"))).unwrap())
                .unwrap();
        let chunks: Vec<Vec<u8>> = chunks.into_iter().map(String::into_bytes).collect();
        let out = relay(&chunks);
        println!(
            "{}",
            json!({"relay_check": n, "events_out": out.len(), "bytes_out": out.iter().map(|b| b.len()).sum::<usize>()})
        );
        measure("relay", &format!("sse_{n}"), iters, || relay(&chunks));
    }
}
