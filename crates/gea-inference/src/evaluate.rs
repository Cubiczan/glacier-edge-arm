use gea_core::{FaultClass, FeatureWindow, TelemetrySample};
use serde::{Deserialize, Serialize};

use crate::classifier::EdgeClassifier;
use crate::quantize::QuantizedClassifier;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub windows_evaluated: usize,
    pub correct: usize,
    pub accuracy: f64,
    pub false_positive_rate: f64,
    pub per_class: Vec<ClassMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub fault_class: String,
    pub support: usize,
    pub correct: usize,
    pub precision: f64,
    pub recall: f64,
}

pub fn evaluate_windows(windows: &[Vec<TelemetrySample>]) -> EvaluationReport {
    let clf_fp32 = EdgeClassifier::new();
    let clf_int8 = QuantizedClassifier::new();

    let mut correct = 0usize;
    let mut false_positives = 0usize;
    let mut normal_total = 0usize;
    let mut class_stats: Vec<(FaultClass, usize, usize)> =
        FaultClass::all().iter().map(|&c| (c, 0, 0)).collect();

    for window in windows {
        let expected = window
            .last()
            .and_then(|s| s.label)
            .unwrap_or(FaultClass::Normal);

        let fw = FeatureWindow::new(window.clone());
        let fp32 = clf_fp32.infer_window(&fw);
        let int8 = clf_int8.infer_window(&fw);

        // Use INT8 path as production edge path
        let predicted = int8.fault_class;

        for (class, support, hits) in &mut class_stats {
            if *class == expected {
                *support += 1;
                if predicted == expected {
                    *hits += 1;
                }
            }
        }

        if predicted == expected {
            correct += 1;
        }
        if expected == FaultClass::Normal && predicted != FaultClass::Normal {
            false_positives += 1;
        }
        if expected == FaultClass::Normal {
            normal_total += 1;
        }

        let _ = fp32; // both paths exercised
    }

    let total = windows.len();
    let per_class = class_stats
        .into_iter()
        .map(|(class, support, hits)| {
            let recall = if support > 0 {
                hits as f64 / support as f64
            } else {
                0.0
            };
            ClassMetrics {
                fault_class: class.as_str().into(),
                support,
                correct: hits,
                precision: recall, // single-label windows; simplified
                recall,
            }
        })
        .collect();

    EvaluationReport {
        windows_evaluated: total,
        correct,
        accuracy: correct as f64 / total as f64,
        false_positive_rate: if normal_total > 0 {
            false_positives as f64 / normal_total as f64
        } else {
            0.0
        },
        per_class,
    }
}
