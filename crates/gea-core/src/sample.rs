use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::FaultClass;

/// Single time-step reading from a BESS rack module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySample {
    pub sample_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub rack_id: String,
    pub module_index: u8,
    pub voltage_v: f32,
    pub current_a: f32,
    pub temperature_c: f32,
    pub impedance_mohm: f32,
    pub state_of_charge_pct: f32,
    pub label: Option<FaultClass>,
}

impl TelemetrySample {
    pub fn power_kw(&self) -> f32 {
        (self.voltage_v * self.current_a) / 1000.0
    }
}

/// A batch of samples for one rack at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RackTelemetry {
    pub rack_id: String,
    pub samples: Vec<TelemetrySample>,
}

impl RackTelemetry {
    pub fn module_count(&self) -> usize {
        self.samples.len()
    }
}
