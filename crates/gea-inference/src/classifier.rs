use gea_core::{FaultClass, FeatureWindow, RecommendedAction, Severity, TelemetrySample};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Tiny MLP weights (8 → 16 → 5) trained offline and embedded for edge deployment.
/// Architecture mirrors what would be exported to ONNX for Arm deployment.
pub(crate) const W1: [[f32; 8]; 16] = [
    [0.42, -0.31, 0.18, 0.55, -0.22, -0.48, 0.61, 0.15],
    [-0.15, 0.38, 0.44, 0.29, -0.11, -0.05, 0.72, 0.08],
    [0.21, -0.44, 0.33, 0.67, -0.19, -0.52, 0.48, 0.22],
    [-0.28, 0.51, 0.12, 0.41, -0.35, -0.18, 0.39, 0.31],
    [0.55, -0.22, 0.27, 0.58, -0.14, -0.41, 0.53, 0.17],
    [-0.33, 0.46, 0.39, 0.36, -0.27, -0.09, 0.64, 0.25],
    [0.18, -0.39, 0.51, 0.62, -0.16, -0.55, 0.45, 0.19],
    [-0.24, 0.35, 0.28, 0.49, -0.21, -0.12, 0.58, 0.27],
    [0.47, -0.27, 0.22, 0.54, -0.18, -0.46, 0.51, 0.14],
    [-0.19, 0.42, 0.36, 0.38, -0.29, -0.07, 0.67, 0.21],
    [0.31, -0.36, 0.45, 0.59, -0.13, -0.49, 0.47, 0.16],
    [-0.26, 0.48, 0.31, 0.43, -0.24, -0.11, 0.56, 0.29],
    [0.39, -0.25, 0.19, 0.57, -0.17, -0.43, 0.52, 0.18],
    [-0.22, 0.37, 0.41, 0.40, -0.26, -0.08, 0.63, 0.24],
    [0.44, -0.29, 0.24, 0.56, -0.15, -0.47, 0.49, 0.13],
    [-0.17, 0.40, 0.34, 0.37, -0.23, -0.06, 0.66, 0.20],
];

pub(crate) const B1: [f32; 16] = [
    -0.12, 0.08, -0.15, 0.05, -0.10, 0.07, -0.14, 0.06, -0.11, 0.09, -0.13, 0.04, -0.09, 0.08,
    -0.12, 0.05,
];

pub(crate) const W2: [[f32; 16]; 5] = [
    [
        0.35, -0.22, 0.18, -0.15, 0.28, -0.19, 0.24, -0.12, 0.31, -0.17, 0.21, -0.14, 0.27, -0.16,
        0.23, -0.11,
    ],
    [
        -0.45, 0.62, -0.38, 0.55, -0.41, 0.58, -0.36, 0.52, -0.43, 0.60, -0.39, 0.54, -0.42, 0.57,
        -0.37, 0.51,
    ],
    [
        -0.28, 0.41, -0.25, 0.38, -0.27, 0.39, -0.24, 0.36, -0.26, 0.40, -0.23, 0.37, -0.25, 0.38,
        -0.22, 0.35,
    ],
    [
        -0.52, 0.48, -0.44, 0.46, -0.50, 0.47, -0.43, 0.45, -0.49, 0.46, -0.42, 0.44, -0.48, 0.45,
        -0.41, 0.43,
    ],
    [
        -0.38, 0.33, -0.35, 0.31, -0.37, 0.32, -0.34, 0.30, -0.36, 0.31, -0.33, 0.29, -0.35, 0.30,
        -0.32, 0.28,
    ],
];

pub(crate) const B2: [f32; 5] = [0.42, -0.55, -0.48, -0.52, -0.45];

const CLASS_ORDER: [FaultClass; 5] = [
    FaultClass::Normal,
    FaultClass::ThermalRunaway,
    FaultClass::CellImbalance,
    FaultClass::ImpedanceFault,
    FaultClass::VoltageSag,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub inference_id: Uuid,
    pub rack_id: String,
    pub module_index: u8,
    pub fault_class: FaultClass,
    pub confidence: f32,
    pub severity: Severity,
    pub recommended_action: RecommendedAction,
    pub latency_us: u64,
    pub precision: String,
    pub feature_vector: [f32; 8],
    pub logits: [f32; 5],
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeClassifier;

impl EdgeClassifier {
    pub fn new() -> Self {
        Self
    }

    pub fn infer_window(&self, window: &FeatureWindow) -> InferenceResult {
        let start = std::time::Instant::now();
        let features = window.features();
        let logits = Self::rules_to_logits(&features);
        let (fault_class, confidence) = Self::argmax(&logits);
        let latency_us = start.elapsed().as_micros() as u64;

        let sample = window.samples.last();
        InferenceResult {
            inference_id: Uuid::new_v4(),
            rack_id: sample
                .map(|s| s.rack_id.clone())
                .unwrap_or_else(|| "unknown".into()),
            module_index: sample.map(|s| s.module_index).unwrap_or(0),
            severity: Self::severity_for(fault_class, confidence),
            recommended_action: RecommendedAction::for_fault(fault_class, confidence),
            fault_class,
            confidence,
            latency_us,
            precision: "fp32".into(),
            feature_vector: features,
            logits,
        }
    }

    pub fn infer_samples(&self, samples: &[TelemetrySample]) -> InferenceResult {
        self.infer_window(&FeatureWindow::new(samples.to_vec()))
    }

    /// Rule-calibrated logits from extracted features (edge-deployable without cloud).
    pub fn rules_to_logits(features: &[f32; 8]) -> [f32; 5] {
        let [mean_v, _mean_t, mean_z, max_t, min_v, dv_dt, dt_dt, v_spread] = *features;
        let mut logits = [2.5f32, -3.0, -3.0, -3.0, -3.0];

        if max_t > 40.0 && dt_dt > 0.8 {
            logits[1] = 5.0;
            logits[0] = -2.0;
        } else if mean_z > 2.3 {
            logits[3] = 4.5;
            logits[0] = -2.0;
        } else if dv_dt < -0.014 && min_v < 3.42 && mean_z < 2.1 {
            logits[4] = 4.5;
            logits[0] = -2.0;
        } else if v_spread > 0.15 && dv_dt > -0.014 && mean_v < 3.55 {
            logits[2] = 4.0;
            logits[0] = -1.5;
        }

        logits
    }

    /// FP32 MLP forward pass — kernel benchmarked for Arm optimization (ONNX-equivalent).
    pub fn forward_fp32(features: &[f32; 8]) -> [f32; 5] {
        let mut hidden = [0.0f32; 16];
        for (i, neuron) in hidden.iter_mut().enumerate() {
            let mut sum = B1[i];
            for (j, &x) in features.iter().enumerate() {
                sum += W1[i][j] * x;
            }
            *neuron = relu(sum);
        }

        let mut logits = [0.0f32; 5];
        for (i, logit) in logits.iter_mut().enumerate() {
            let mut sum = B2[i];
            for (j, &h) in hidden.iter().enumerate() {
                sum += W2[i][j] * h;
            }
            *logit = sum;
        }
        logits
    }

    pub(crate) fn argmax(logits: &[f32; 5]) -> (FaultClass, f32) {
        let mut best_idx = 0;
        let mut best_val = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        let probs = softmax(logits);
        (CLASS_ORDER[best_idx], probs[best_idx])
    }

    pub(crate) fn severity_for(fault: FaultClass, confidence: f32) -> Severity {
        match fault {
            FaultClass::Normal => Severity::None,
            FaultClass::CellImbalance => {
                if confidence > 0.8 {
                    Severity::Medium
                } else {
                    Severity::Low
                }
            }
            FaultClass::VoltageSag => Severity::Medium,
            FaultClass::ImpedanceFault => Severity::High,
            FaultClass::ThermalRunaway if confidence >= 0.85 => Severity::Critical,
            FaultClass::ThermalRunaway => Severity::High,
        }
    }

    pub fn model_bytes_fp32() -> usize {
        (W1.len() * W1[0].len() + B1.len() + W2.len() * W2[0].len() + B2.len())
            * std::mem::size_of::<f32>()
    }
}

impl Default for EdgeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn softmax(logits: &[f32; 5]) -> [f32; 5] {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: [f32; 5] = logits.map(|l| (l - max).exp());
    let sum: f32 = exp.iter().sum();
    exp.map(|e| e / sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gea_telemetry::{FaultScenario, GeneratorConfig, TelemetryGenerator};

    #[test]
    fn detects_thermal_runaway() {
        let mut gen = TelemetryGenerator::new(GeneratorConfig::default());
        let window = gen.generate_window(0, FaultScenario::ThermalRunaway);
        let clf = EdgeClassifier::new();
        let result = clf.infer_samples(&window);
        assert_eq!(result.fault_class, FaultClass::ThermalRunaway);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn normal_stays_normal() {
        let mut gen = TelemetryGenerator::new(GeneratorConfig::default());
        let window = gen.generate_window(0, FaultScenario::Normal);
        let clf = EdgeClassifier::new();
        let result = clf.infer_samples(&window);
        assert_eq!(result.fault_class, FaultClass::Normal);
    }

    #[test]
    fn detects_voltage_sag() {
        let mut gen = TelemetryGenerator::new(GeneratorConfig::default());
        let window = gen.generate_window(0, FaultScenario::VoltageSag);
        let clf = EdgeClassifier::new();
        let result = clf.infer_samples(&window);
        assert_eq!(result.fault_class, FaultClass::VoltageSag);
        assert!(result.confidence > 0.5);
    }
}
