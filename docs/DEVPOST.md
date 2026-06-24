# Devpost Submission — GlacierEdge-Arm

**Challenge:** [Arm AI Optimization Challenge 2026](https://arm-ai-optimization-challenge.devpost.com/)  
**Track:** Physical AI (Track 1 — optimization output)  
**Repos:** [Codeberg](https://codeberg.org/cubiczan/glacier-edge-arm) · [GitHub](https://github.com/Cubiczan/glacier-edge-arm)

---

## General info (copy-paste)

### Project name

```
GlacierEdge-Arm
```

### Elevator pitch (200 char max)

```
Sub-50ms Arm edge inference that catches BESS faults—thermal runaway, imbalance, impedance—before data center backup power fails. Open-source Rust, INT8 quantized, CHP-audited.
```

*Character count: 168*

### Alternate elevator pitches

**Shorter (tagline):**
```
Arm-optimized edge AI that guards data center backup batteries before they fail.
```

**Impact-focused:**
```
Open-source Rust edge stack detecting BESS faults in under 50ms on Arm—3× smaller INT8 models, signed audit trails, built for Physical AI.
```

---

## Project overview (next Devpost section)

GlacierEdge-Arm is an open-source edge inference stack for battery energy storage systems (BESS) backing data center power. AI workloads are driving unprecedented power demand; when backup batteries fail, downtime costs thousands per minute. GlacierEdge-Arm runs on Arm edge hardware (Raspberry Pi 5, Graviton gateways) and classifies five fault types—thermal runaway, cell imbalance, impedance fault, voltage sag, and normal—from live cell telemetry in sub-millisecond time.

The project demonstrates measurable Arm optimization: INT8 quantization reduces model size by **3.1×** (916 → 292 bytes) while maintaining **96%+ accuracy** on synthetic BESS windows. Every inference is logged to a CHP-governed, HMAC-signed audit ledger suitable for safety-critical energy infrastructure.

Built entirely under the **cubiczan** open-source brand in Rust with MIT license.

---

## Functionality / output

- **Synthetic BESS telemetry generator** — 5 labeled fault scenarios for testing without hardware
- **8-dimensional feature extraction** — voltage, temperature, impedance, derivatives, spread
- **Dual inference kernels** — FP32 and INT8-quantized 8→16→5 MLP (ONNX-equivalent)
- **Edge classification** — fault class, confidence, severity, recommended action (derate, isolate, trip)
- **CHP governance** — signed JSONL audit trail per inference
- **Benchmark harness** — reproducible FP32 vs INT8 latency and model size comparison

**Demo command:**
```bash
cargo run -p gea-cli -- demo --scenario thermal_runaway
```

---

## Setup instructions (judges)

```bash
git clone https://codeberg.org/cubiczan/glacier-edge-arm.git
cd glacier-edge-arm
cargo build --release -p gea-cli

# Demo fault detection
./target/release/gea demo --scenario thermal_runaway

# Evaluate accuracy
./target/release/gea evaluate --windows-per-scenario 100

# Benchmark Arm optimization
./target/release/gea bench --iterations 10000 --json
```

**Requirements:** Rust 1.70+ (stable). Runs natively on aarch64 (Apple Silicon, Graviton, Pi 5).

---

## Built with

- Rust / Cargo workspace
- Arm aarch64 (native + cross-compile to Pi 5)
- INT8 quantization (ONNX-equivalent kernel)
- Consensus Hardening Protocol (CHP) governance patterns

---

## Try it out links

| Link | URL |
|------|-----|
| Codeberg repo | https://codeberg.org/cubiczan/glacier-edge-arm |
| GitHub repo | https://github.com/Cubiczan/glacier-edge-arm |
| README | https://codeberg.org/cubiczan/glacier-edge-arm/src/branch/main/README.md |

---

## Video script (under 3 min)

1. **Problem (20s):** Data centers need reliable backup power; BESS faults cause costly downtime.
2. **Solution (30s):** GlacierEdge-Arm — edge Arm inference on cell telemetry.
3. **Live demo (60s):** `gea demo --scenario thermal_runaway` → detection, TripBreaker action, audit log.
4. **Benchmark (30s):** `gea bench` → 3.1× size reduction, sub-ms latency on aarch64.
5. **Evaluate (20s):** 99.8% accuracy, 0% false positives on normal.
6. **Close (20s):** Open source, cubiczan, Physical AI track, Arm Performix-ready.

---

## Checklist before submit

- [x] Public MIT repo on Codeberg + GitHub
- [x] Open source license visible in repo
- [x] Demo video in repo (`assets/demo.mp4`) — upload to YouTube/Vimeo for Devpost
- [ ] Arm Performix benchmark on Graviton or Pi 5 (optional but strong)
- [x] Thumbnail image for Devpost (`assets/thumbnail.png`)
- [ ] Select track: **Physical AI**
