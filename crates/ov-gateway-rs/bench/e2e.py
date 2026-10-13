"""End-to-end load: Python gateway (repo source) vs ov-gateway-rs under agent_sim.

Usage:
  python e2e.py mocks start|stop              # mock OV (core 4) + N mock upstreams (cores 2-4)
  python e2e.py run --gateway py|rs --sessions N [--warmup 60 --duration 120 --ramp 30
                    --shards K --label L]
Environment: E2E_ROOT is the scratch root holding pyvenv/, bench/ (agent_sim.py,
mock_agent_upstream.py, mock_ov.py copied from the w157 harness) and run/.

The gateway is started on empty storage, pinned to GW_CORE (default 1). Gateway CPU is read from
/proc/<pid>/stat (utime+stime of the process and its children, all threads) at the start
and end of the measurement window; CPU/request = delta / requests started in the window.
Mocks and simulator CPU are reported so a saturated load generator is visible.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.environ["E2E_ROOT"]
CRATE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = f"{ROOT}/pyvenv/bin/python"
TICK = os.sysconf("SC_CLK_TCK")
PORTS = {"mock": 23817, "ov": 23918, "py": 23925, "rs": 23955}
GW_CORE = os.environ.get("GW_CORE", "1")
ENV = {"HOME": f"{ROOT}/home", "PATH": f"{ROOT}/pyvenv/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}
OV_KEY = "ov-user-key-alice"
RS_KEY = "ovgw_rust_bench_key"


def spawn(cmd, log, cores, env=None):
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    return subprocess.Popen(
        ["taskset", "-c", cores, *cmd], stdout=open(f"{ROOT}/logs/{log}", "w"),
        stderr=subprocess.STDOUT, env={**ENV, **(env or {})}, cwd=ROOT, start_new_session=True)


def pids(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def cpu_seconds(pid, part=None):
    """utime+stime (or just "user"/"sys") of pid and its live descendants, all threads."""
    total = 0.0
    todo = [pid]
    while todo:
        p = todo.pop()
        try:
            fields = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()
            user, system = int(fields[11]) / TICK, int(fields[12]) / TICK
            total += {"user": user, "sys": system}.get(part, user + system)
            for tid in os.listdir(f"/proc/{p}/task"):
                kids = open(f"/proc/{p}/task/{tid}/children").read().split()
                todo.extend(int(k) for k in kids)
        except OSError:
            pass
    return total


def core_busy_seconds(core=0):
    """Busy time (all tasks) of one core, from /proc/stat."""
    for line in open("/proc/stat"):
        if line.startswith(f"cpu{core} "):
            v = [int(x) for x in line.split()[1:]]
            return (sum(v) - v[3] - v[4]) / TICK
    return 0.0


def rss_mb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        return 0
    return 0


def wait_http(url, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return True
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("not ready: " + url)


def call(method, path, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORTS['py']}/gateway{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"x-api-key": OV_KEY, "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def mocks(action, upstreams=2):
    for pattern in ("bench/mock_agent_upstream.py", "bench/mock_ov.py"):
        for p in pids(pattern):
            os.kill(p, signal.SIGTERM)
    time.sleep(1)
    if action == "stop":
        return
    # A sibling benchmark on this host pins its gateway to core 0 and its load to 4-7.
    spawn([PY, "bench/mock_ov.py"], "mock_ov.log", "4", {"MOCK_OV_PORT": str(PORTS["ov"])})
    env = {"MOCK_PORT": str(PORTS["mock"]), "MOCK_REUSE": "1", "MOCK_TTFT_S": "2.0", "MOCK_TPS": "60",
           "MOCK_OUT_MIN": "200", "MOCK_OUT_MAX": "1500", "MOCK_P_TOOL": "0.8", "MOCK_P_OV": "0.05"}
    for i in range(upstreams):
        spawn([PY, "bench/mock_agent_upstream.py"], f"mock_up{i}.log", ["2", "3", "4"][i % 3], env)
    wait_http(f"http://127.0.0.1:{PORTS['mock']}/stats")
    print("mocks up", pids("bench/mock_agent_upstream.py"), pids("bench/mock_ov.py"))


def start_gateway(kind):
    # tmpfs: the shared host has heavy disk/swap pressure; fsync noise would dominate.
    store = f"/dev/shm/ovgw-bench-{kind}"
    subprocess.run(["rm", "-rf", store])
    os.makedirs(store)
    if kind == "py":
        conf = f"{ROOT}/run/pygw.conf"
        json.dump({"openviking_url": f"http://127.0.0.1:{PORTS['ov']}", "host": "127.0.0.1",
                   "port": PORTS["py"], "workers": 1, "storage_path": store}, open(conf, "w"))
        proc = spawn([f"{ROOT}/pyvenv/bin/openviking-gateway", "--config", conf], "pygw.log", GW_CORE)
        wait_http(f"http://127.0.0.1:{PORTS['py']}/health")
        call("PUT", "/admin/upstreams/mock-anthropic", {
            "name": "mock-anthropic", "protocol": "anthropic", "base_url": f"http://127.0.0.1:{PORTS['mock']}",
            "api_key": "mock-upstream-key", "models": ["mock-model"], "vendor": "generic"})
        call("PUT", "/admin/policies/default", {"name": "default"})
        key = call("POST", "/admin/keys", {"name": "bench", "openviking_key": OV_KEY, "policy_id": "default",
                                           "upstream_ids": ["mock-anthropic"]})["key"]
        return proc, key, f"http://127.0.0.1:{PORTS['py']}/gateway"
    conf = f"{ROOT}/run/rsgw.json"
    json.dump({"listen": f"127.0.0.1:{PORTS['rs']}", "openviking_url": f"http://127.0.0.1:{PORTS['ov']}",
               "storage_path": store, "upstream_base": f"http://127.0.0.1:{PORTS['mock']}",
               "upstream_key": "mock-upstream-key",
               "keys": {RS_KEY: {"account": "bench", "user_id": "alice", "openviking_key": OV_KEY}}},
              open(conf, "w"))
    proc = spawn([f"{CRATE}/target/release/ov-gateway-rs", conf], "rsgw.log", GW_CORE)
    wait_http(f"http://127.0.0.1:{PORTS['rs']}/health")
    return proc, RS_KEY, f"http://127.0.0.1:{PORTS['rs']}/gateway"


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    return round(v[min(len(v) - 1, int(round((len(v) - 1) * p / 100)))], 1)


def wait_quiet_core(limit=600, threshold=10.0):
    """Wait until other tenants use < threshold % of the gateway core (nothing of ours runs there yet)."""
    t0 = time.time()
    while True:
        k0 = core_busy_seconds(int(GW_CORE))
        time.sleep(5)
        busy = (core_busy_seconds(int(GW_CORE)) - k0) / 5 * 100
        if busy < threshold or time.time() - t0 > limit:
            return round(time.time() - t0), round(busy, 1)
        time.sleep(10)


def run(args):
    out = f"{ROOT}/results/{args.label}"
    os.makedirs(out, exist_ok=True)
    waited, idle_busy = wait_quiet_core(limit=args.quiet_wait)
    print(f"core {GW_CORE} quiet after {waited}s (busy {idle_busy}%)", flush=True)
    proc, key, url = start_gateway(args.gateway)
    gw = proc.pid
    shards = []
    per = args.sessions // args.shards
    sim_cores = ["5", "6", "7", "3"]
    for i in range(args.shards):
        raw = f"{out}/shard{i}.jsonl"
        if os.path.exists(raw):
            os.remove(raw)
        shards.append(spawn(
            [PY, "bench/agent_sim.py", "--sessions", str(per), "--warmup", str(args.warmup),
             "--duration", str(args.duration), "--ramp", str(args.ramp), "--profile", "default",
             "--gateway", url, "--key", key, "--seed", str(i + 1), "--raw", raw, "--label", args.label],
            f"sim_{args.label}_{i}.log", sim_cores[i % len(sim_cores)]))
    t0 = time.time()
    others = {"mock_upstream": pids("bench/mock_agent_upstream.py"), "mock_ov": pids("bench/mock_ov.py"),
              "sim": [s.pid for s in shards]}
    time.sleep(args.warmup)
    load_start = os.getloadavg()[0]
    c0 = cpu_seconds(gw)
    u0 = cpu_seconds(gw, "user")
    k0 = core_busy_seconds(int(GW_CORE))
    o0 = {k: sum(cpu_seconds(p) for p in v) for k, v in others.items()}
    samples = []
    t_start = time.time()
    while time.time() - t_start < args.duration:
        time.sleep(5)
        samples.append({"t": round(time.time() - t_start, 1), "gw_cpu_s": cpu_seconds(gw), "rss_mb": rss_mb(gw)})
    c1 = cpu_seconds(gw)
    u1 = cpu_seconds(gw, "user")
    k1 = core_busy_seconds(int(GW_CORE))
    o1 = {k: sum(cpu_seconds(p) for p in v) for k, v in others.items()}
    window = time.time() - t_start
    load_end = os.getloadavg()[0]
    for s in shards:
        s.wait()
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    rows = []
    for i in range(args.shards):
        rows.extend(json.loads(line) for line in open(f"{out}/shard{i}.jsonl"))
    m = [r for r in rows if r.get("measured")]
    ok = [r for r in m if not r.get("error")]
    summary = {
        "label": args.label, "gateway": args.gateway, "sessions": args.sessions, "shards": args.shards,
        "window_s": round(window, 1), "requests": len(m), "errors": len(m) - len(ok),
        "error_kinds": {k: sum(1 for r in m if r.get("error") == k) for k in {r.get("error") for r in m} if k},
        "rps": round(len(m) / window, 2),
        "gw_cpu_pct": round((c1 - c0) / window * 100, 1),
        "gw_cpu_ms_per_req": round((c1 - c0) * 1000 / max(1, len(m)), 2),
        "gw_user_ms_per_req": round((u1 - u0) * 1000 / max(1, len(m)), 2),
        "gw_sys_ms_per_req": round(((c1 - c0) - (u1 - u0)) * 1000 / max(1, len(m)), 2),
        # Core 0 time used by anything other than the gateway: >10% means a noisy neighbour.
        "core_other_pct": round(((k1 - k0) - (c1 - c0)) / window * 100, 1),
        "gw_rss_mb_max": round(max((s["rss_mb"] for s in samples), default=0), 1),
        "other_cpu_pct": {k: round((o1[k] - o0[k]) / window * 100, 1) for k in o0},
        "loadavg": [round(load_start, 2), round(load_end, 2)],
        "user_turn_share": round(sum(1 for r in ok if r["kind"] == "user") / max(1, len(ok)), 3),
        "hidden_share": round(sum(1 for r in ok if r.get("hidden")) / max(1, len(ok)), 3),
        "body_kb_p50": pct([r["bytes"] / 1024 for r in ok], 50),
    }
    for key_ in ("added_ttft_ms", "added_pre_ms"):
        for kind in ("continuation", "user"):
            v = [r[key_] for r in ok if key_ in r and r["kind"] == kind and not r.get("hidden")]
            summary[f"{key_}.{kind}"] = {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99)}
    json.dump({"summary": summary, "samples": samples}, open(f"{out}/summary.json", "w"), indent=1)
    print(json.dumps(summary), flush=True)
    return summary


def run_clean(args, attempts=3, max_noise=15.0):
    """Rerun a point while another tenant took > max_noise % of core 0; keep the cleanest."""
    label, best = args.label, None
    for attempt in range(attempts):
        args.label = f"{label}_try{attempt}"
        summary = run(args)
        if best is None or summary["core_other_pct"] < best[1]["core_other_pct"]:
            best = (args.label, summary)
        if summary["core_other_pct"] <= max_noise:
            break
    subprocess.run(["rm", "-rf", f"{ROOT}/results/{label}"])
    subprocess.run(["cp", "-r", f"{ROOT}/results/{best[0]}", f"{ROOT}/results/{label}"])
    print(f"kept {best[0]} (gateway-core noise {best[1]['core_other_pct']}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("action", nargs="?")
    ap.add_argument("--gateway", choices=["py", "rs"])
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--warmup", type=float, default=60)
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--ramp", type=float, default=30)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--upstreams", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--quiet-wait", type=float, default=600)
    args = ap.parse_args()
    if args.cmd == "mocks":
        mocks(args.action, args.upstreams)
    elif args.cmd == "idle":
        proc, _, _ = start_gateway(args.gateway)
        time.sleep(3)
        c0 = cpu_seconds(proc.pid)
        time.sleep(args.duration)
        c1 = cpu_seconds(proc.pid)
        proc.terminate()
        proc.wait()
        print(json.dumps({"gateway": args.gateway, "idle_cpu_pct": round((c1 - c0) / args.duration * 100, 2)}))
    elif args.cmd == "run":
        args.label = args.label or f"{args.gateway}_n{args.sessions}"
        run_clean(args, attempts=args.attempts)
    else:
        sys.exit("unknown command")


if __name__ == "__main__":
    main()
