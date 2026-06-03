# Benchmark results

This directory holds JSON + Markdown output from
``python -m benchmark.run_all`` and the individual ``bench_*`` scripts.

Layout:

* ``attention.json``         : per-config attention timings.
* ``kv_cache.json``          : paged KV cache append + lookup timings.
* ``fused_linear_norm.json`` : fused Linear+Norm timings.
* ``softmax.json``           : online softmax timings.
* ``rope.json``              : RoPE timings.
* ``summary.json``           : combined output of all the above.
* ``summary.md``             : human-readable summary table.

Each JSON file records the active device, compute capability, and torch
version so a result can be reproduced.

Re-run with:

```bash
python -m benchmark.run_all --warmup 10 --iters 100
```

For a smoke run that finishes in <30s on an A100:

```bash
python -m benchmark.run_all --warmup 2 --iters 5
```

The committed results in this directory are placeholders (empty or
device-specific snapshots). CI does not commit benchmark output; run the
harness on your target device to produce comparable numbers.
