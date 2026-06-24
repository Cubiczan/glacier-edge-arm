use gea_core::FeatureWindow;
use gea_telemetry::{GeneratorConfig, TelemetryGenerator};
use serde::{Deserialize, Serialize};

use crate::classifier::EdgeClassifier;
use crate::quantize::QuantizedClassifier;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    Fp32,
    Int8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub precision: Precision,
    pub iterations: u64,
    pub total_us: u64,
    pub mean_us: f64,
    pub p50_us: u64,
    pub p99_us: u64,
    pub model_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub platform: String,
    pub fp32: BenchResult,
    pub int8: BenchResult,
    pub speedup_ratio: f64,
    pub size_reduction_ratio: f64,
}

pub fn run_benchmark(iterations: u64) -> BenchReport {
    let mut gen = TelemetryGenerator::new(GeneratorConfig::default());
    let window = gen.generate_window(0, gea_telemetry::FaultScenario::Normal);
    let feature_window = FeatureWindow::new(window);
    let features = feature_window.features();

    let fp32 = bench_fp32(&features, iterations);
    let int8 = bench_int8(&features, iterations);

    BenchReport {
        platform: std::env::consts::ARCH.into(),
        speedup_ratio: fp32.mean_us / int8.mean_us.max(1.0),
        size_reduction_ratio: fp32.model_bytes as f64 / int8.model_bytes as f64,
        fp32,
        int8,
    }
}

fn bench_fp32(features: &[f32; 8], iterations: u64) -> BenchResult {
    let mut latencies = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let _ = EdgeClassifier::forward_fp32(features);
        latencies.push(start.elapsed().as_nanos() as u64);
    }
    summarize(
        Precision::Fp32,
        latencies,
        EdgeClassifier::model_bytes_fp32(),
    )
}

fn bench_int8(features: &[f32; 8], iterations: u64) -> BenchResult {
    let mut latencies = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let _ = QuantizedClassifier::forward_int8(features);
        latencies.push(start.elapsed().as_nanos() as u64);
    }
    summarize(
        Precision::Int8,
        latencies,
        QuantizedClassifier::model_bytes_int8(),
    )
}

fn summarize(precision: Precision, mut latencies: Vec<u64>, model_bytes: usize) -> BenchResult {
    latencies.sort_unstable();
    let iterations = latencies.len() as u64;
    let total_ns: u64 = latencies.iter().sum();
    let mean_us = total_ns as f64 / iterations as f64 / 1000.0;
    let p50_us = (latencies[latencies.len() / 2] as f64 / 1000.0) as u64;
    let p99_idx = ((latencies.len() as f64) * 0.99).floor() as usize;
    let p99_us = (latencies[p99_idx.min(latencies.len() - 1)] as f64 / 1000.0) as u64;

    BenchResult {
        precision,
        iterations,
        total_us: (total_ns / 1000) as u64,
        mean_us,
        p50_us: p50_us.max(1),
        p99_us: p99_us.max(1),
        model_bytes,
    }
}
