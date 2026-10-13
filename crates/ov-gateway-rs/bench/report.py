"""Markdown tables from e2e.py summaries and the microbenchmark JSON lines.

Usage: python report.py RESULTS_DIR LABEL [LABEL ...]   (e2e table)
       python report.py --micro PY.jsonl RS.jsonl        (microbench table)
"""

import json
import sys


def e2e(root, labels):
    print("| gateway | sessions | req/s | CPU % of 1 core | CPU ms/req (user+sys) | "
          "tool-result turn extra TTFT p50/p95/p99 ms | user turn p50/p95/p99 ms | errors | "
          "RSS MB | gateway-core noise % | sim / mock CPU % |")
    print("|-|-|-|-|-|-|-|-|-|-|-|")
    for label in labels:
        s = json.load(open(f"{root}/{label}/summary.json"))["summary"]
        c, u = s["added_ttft_ms.continuation"], s["added_ttft_ms.user"]
        print(f"| {s['gateway']} | {s['sessions']} | {s['rps']} | {s['gw_cpu_pct']} | "
              f"{s['gw_cpu_ms_per_req']} ({s.get('gw_user_ms_per_req', '?')}+{s.get('gw_sys_ms_per_req', '?')}) | "
              f"{c['p50']}/{c['p95']}/{c['p99']} | {u['p50']}/{u['p95']}/{u['p99']} | {s['errors']} | "
              f"{s['gw_rss_mb_max']} | {s.get('core_other_pct', s.get('core0_other_pct', 'n/a'))} | "
              f"{s['other_cpu_pct']['sim']} / {s['other_cpu_pct']['mock_upstream']} |")


def micro(py_path, rs_path):
    def load(path):
        return {(r["name"], r["fixture"]): r["cpu_ms"]
                for r in map(json.loads, filter(lambda l: l.startswith("{"), open(path)))
                if "name" in r}

    py, rs = load(py_path), load(rs_path)
    print("| step | input | Python ms | Rust ms | speedup |")
    print("|-|-|-|-|-|")
    for key, value in py.items():
        if key in rs:
            print(f"| {key[0]} | {key[1]} | {value:.3f} | {rs[key]:.3f} | {value / rs[key]:.1f}x |")


if __name__ == "__main__":
    if sys.argv[1] == "--micro":
        micro(sys.argv[2], sys.argv[3])
    else:
        e2e(sys.argv[1], sys.argv[2:])
