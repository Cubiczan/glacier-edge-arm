# GlacierEdge-Arm

**Arm-optimized edge inference for data center BESS fault detection.**

GlacierEdge-Arm is a cubiczan open-source project for the [Arm AI Optimization Challenge 2026](https://arm-ai-optimization-challenge.devpost.com/). It detects battery energy storage system (BESS) faults at the edge — thermal runaway, cell imbalance, impedance faults, and voltage sag — with sub-millisecond inference, INT8 quantization, and CHP-governed audit trails.

Built for **Physical AI**: embedded Arm controllers monitoring data center backup power.

## Features

- **Synthetic BESS telemetry** — chemistry-agnostic cell/rack windows with labeled fault scenarios
- **8-dimensional feature extraction** — voltage, temperature, impedance, derivatives, spread
- **Dual inference paths** — FP32 and INT8-quantized MLP kernels (ONNX-equivalent structure)
- **Rule-calibrated classification** — edge-deployable without cloud connectivity
- **CHP governance** — signed JSONL audit ledger for every inference
- **Benchmark harness** — reproducible FP32 vs INT8 latency and model size comparison

## Quick start

```bash
git clone https://codeberg.org/cubiczan/glacier-edge-arm.git
cd glacier-edge-arm
cargo build --release -p gea-cli

# Demo: inject thermal runaway and detect
cargo run -p gea-cli -- demo --scenario thermal_runaway

# Evaluate accuracy on synthetic dataset
cargo run -p gea-cli -- evaluate --windows-per-scenario 100

# Benchmark FP32 vs INT8 kernels
cargo run -p gea-cli -- bench --iterations 10000

# Generate telemetry JSON
cargo run -p gea-cli -- generate --scenario impedance_fault --output data/fault.json

# Run inference on telemetry file
cargo run -p gea-cli -- infer --input data/fault.json --precision int8
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    gea-telemetry                             │
│  Synthetic BESS windows: normal, thermal, imbalance, ...    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    gea-core                                  │
│  FeatureWindow → 8D vector (V, T, Z, dV/dt, dT/dt, ...)    │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐   ┌──────────────────────┐
│  gea-inference FP32  │   │  gea-inference INT8  │
│  MLP 8→16→5 kernel   │   │  Quantized matmul    │
└──────────┬───────────┘   └──────────┬───────────┘
           │                          │
           └────────────┬─────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Fault class + confidence + recommended action               │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  gea-governance — CHP audit ledger (HMAC-SHA256 signed)     │
└─────────────────────────────────────────────────────────────┘
```

## Crate layout

| Crate | Role |
|-------|------|
| `gea-core` | Domain types: `TelemetrySample`, `FaultClass`, `FeatureWindow` |
| `gea-telemetry` | Synthetic BESS data generator with 5 fault scenarios |
| `gea-inference` | FP32/INT8 classifiers, benchmarks, evaluation |
| `gea-governance` | CHP policy + signed audit ledger |
| `gea-cli` | `gea` binary — generate, infer, bench, evaluate, demo, audit |

## Fault scenarios

| Scenario | Signature | Action |
|----------|-----------|--------|
| `normal` | Stable V/T/Z | Monitor |
| `thermal_runaway` | Rising temp + falling voltage | Trip breaker |
| `cell_imbalance` | Voltage spread across modules | Isolate module |
| `impedance_fault` | Rising internal impedance | Isolate module |
| `voltage_sag` | Sustained voltage drop | Derate |

## Arm optimization

Target metrics for Devpost submission:

| Metric | FP32 | INT8 |
|--------|------|------|
| Model size | ~916 bytes | ~292 bytes |
| Mean latency | benchmark on your hardware | benchmark on your hardware |
| Size reduction | — | ~3.1× |

Run on **Raspberry Pi 5** (edge proxy) or **AWS Graviton** (rack gateway):

```bash
cargo build --release -p gea-cli
./target/release/gea bench --iterations 50000 --json > benchmarks/latest.json
```

See [benchmarks/README.md](benchmarks/README.md) for Arm Performix integration.

## CHP governance

Every inference is logged to `.gea/audit.jsonl` with:

- Fault class, confidence, severity, recommended action
- Latency and precision path (fp32 / int8)
- Content hash + optional HMAC signature (`GEA_SIGNING_KEY`)

```bash
export GEA_SIGNING_KEY=your-dev-key
cargo run -p gea-cli -- demo --scenario thermal_runaway
cargo run -p gea-cli -- audit
```

## Optional: train and export ONNX

```bash
cd tools
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_model.py --output ../models/bess_fault.onnx
```

## Tests

```bash
cargo test --all
```

## License

MIT — see [LICENSE](LICENSE).

## Author

[cubiczan](https://codeberg.org/cubiczan) — Arm AI Optimization Challenge 2026 submission.
