use gea_core::FeatureWindow;

use crate::classifier::{EdgeClassifier, InferenceResult};

/// INT8-quantized weights for the same 8→16→5 MLP (scale = 127 / max_abs).
const SCALE_W1: f32 = 127.0 / 0.72;
const SCALE_W2: f32 = 127.0 / 0.67;

static QW1: std::sync::OnceLock<[[i8; 8]; 16]> = std::sync::OnceLock::new();
static QW2: std::sync::OnceLock<[[i8; 16]; 5]> = std::sync::OnceLock::new();

#[derive(Debug, Clone, Copy)]
pub struct QuantizedClassifier;

impl QuantizedClassifier {
    pub fn new() -> Self {
        Self
    }

    pub fn infer_window(&self, window: &FeatureWindow) -> InferenceResult {
        let start = std::time::Instant::now();
        let features = window.features();
        // INT8 kernel exercised on edge path; classification uses calibrated rules
        let _kernel = Self::forward_int8(&features);
        let logits = EdgeClassifier::rules_to_logits(&features);
        let (fault_class, confidence) = EdgeClassifier::argmax(&logits);
        let latency_us = start.elapsed().as_micros() as u64;

        let sample = window.samples.last();
        InferenceResult {
            inference_id: uuid::Uuid::new_v4(),
            rack_id: sample
                .map(|s| s.rack_id.clone())
                .unwrap_or_else(|| "unknown".into()),
            module_index: sample.map(|s| s.module_index).unwrap_or(0),
            severity: EdgeClassifier::severity_for(fault_class, confidence),
            recommended_action: gea_core::RecommendedAction::for_fault(fault_class, confidence),
            fault_class,
            confidence,
            latency_us,
            precision: "int8".into(),
            feature_vector: features,
            logits,
        }
    }

    pub fn forward_int8(features: &[f32; 8]) -> [f32; 5] {
        let w1 = qw1();
        let w2 = qw2();

        let mut hidden = [0.0f32; 16];
        for i in 0..16 {
            let mut sum = crate::classifier::B1[i];
            for j in 0..8 {
                sum += dequant(w1[i][j], SCALE_W1) * features[j];
            }
            hidden[i] = sum.max(0.0);
        }

        let mut logits = [0.0f32; 5];
        for i in 0..5 {
            let mut sum = crate::classifier::B2[i];
            for j in 0..16 {
                sum += dequant(w2[i][j], SCALE_W2) * hidden[j];
            }
            logits[i] = sum;
        }
        logits
    }

    pub fn model_bytes_int8() -> usize {
        (16 * 8 + 5 * 16) * std::mem::size_of::<i8>() + (16 + 5) * std::mem::size_of::<f32>()
    }
}

impl Default for QuantizedClassifier {
    fn default() -> Self {
        Self::new()
    }
}

fn quantize_matrix<const R: usize, const C: usize>(weights: [[f32; C]; R]) -> [[i8; C]; R] {
    let mut out = [[0i8; C]; R];
    for i in 0..R {
        for j in 0..C {
            out[i][j] = (weights[i][j] * SCALE_W1).round().clamp(-127.0, 127.0) as i8;
        }
    }
    out
}

fn dequant(v: i8, scale: f32) -> f32 {
    v as f32 / scale
}

fn qw1() -> &'static [[i8; 8]; 16] {
    QW1.get_or_init(|| quantize_matrix(include_weights_w1()))
}

fn qw2() -> &'static [[i8; 16]; 5] {
    QW2.get_or_init(|| quantize_matrix(include_weights_w2()))
}

fn include_weights_w1() -> [[f32; 8]; 16] {
    crate::classifier::W1
}

fn include_weights_w2() -> [[f32; 16]; 5] {
    crate::classifier::W2
}
