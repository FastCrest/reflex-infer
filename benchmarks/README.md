# Benchmarks

This folder is reserved for benchmark scripts and raw result summaries.

Planned files:

- `run_microbench.py`: operator-level benchmark runner.
- `run_decode_loop.py`: end-to-end token generation benchmark.
- `collect_tegrastats.sh`: Jetson power, clock, memory, and thermal logging.
- `results/`: raw benchmark CSV/JSON outputs.
- `profiles/`: Nsight Systems and Nsight Compute reports.

No benchmark numbers should be committed without the corresponding environment
metadata and command line.

