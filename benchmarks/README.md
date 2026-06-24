# GlacierEdge-Arm Benchmarks

Reproducible performance artifacts for the [Arm AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/).

## Quick benchmark

```bash
cargo run -p gea-cli -- bench --iterations 10000
cargo run -p gea-cli -- bench --iterations 10000 --json > benchmarks/latest.json
```

## Arm Performix (Graviton / Neoverse)

On an Arm64 cloud instance (AWS Graviton3, Azure Cobalt, etc.):

```bash
# Build release binary
cargo build --release -p gea-cli

# Run kernel benchmark
./target/release/gea bench --iterations 50000 --json > benchmarks/graviton.json

# Profile with Arm Performix (install from Arm Developer)
# https://developer.arm.com/servers-and-cloud-computing/arm-performix
performix record ./target/release/gea bench --iterations 10000
```

## Raspberry Pi 5 (edge proxy)

```bash
cargo build --release -p gea-cli --target aarch64-unknown-linux-gnu
./target/aarch64-unknown-linux-gnu/release/gea demo --scenario thermal_runaway
./target/aarch64-unknown-linux-gnu/release/gea bench --iterations 10000
```

## Evaluation accuracy

```bash
cargo run -p gea-cli -- evaluate --windows-per-scenario 100
```

Target: **>95% accuracy**, **<5% false positive rate** on synthetic BESS windows.

## Metrics to capture for Devpost

| Metric | FP32 | INT8 |
|--------|------|------|
| Mean inference latency (µs) | | |
| P99 latency (µs) | | |
| Model size (bytes) | | |
| End-to-end fault detection (demo) | | |
