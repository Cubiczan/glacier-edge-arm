use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChpPolicy {
    pub protocol: String,
    pub version: String,
    pub domain: String,
    pub foundation_threshold: u8,
    pub require_adversarial_review: bool,
    pub inference_confidence_floor: f32,
    pub critical_fault_classes: Vec<String>,
}

impl Default for ChpPolicy {
    fn default() -> Self {
        Self {
            protocol: "consensus-hardening-protocol".into(),
            version: "0.1.0".into(),
            domain: "bess_edge_inference".into(),
            foundation_threshold: 75,
            require_adversarial_review: true,
            inference_confidence_floor: 0.55,
            critical_fault_classes: vec!["thermal_runaway".into()],
        }
    }
}

impl ChpPolicy {
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_yaml::from_str(&text).map_err(|e| e.to_string())
    }
}
