use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultClass {
    Normal,
    ThermalRunaway,
    CellImbalance,
    ImpedanceFault,
    VoltageSag,
}

impl FaultClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::ThermalRunaway => "thermal_runaway",
            Self::CellImbalance => "cell_imbalance",
            Self::ImpedanceFault => "impedance_fault",
            Self::VoltageSag => "voltage_sag",
        }
    }

    pub fn is_fault(self) -> bool {
        !matches!(self, Self::Normal)
    }

    pub fn all() -> &'static [FaultClass] {
        &[
            Self::Normal,
            Self::ThermalRunaway,
            Self::CellImbalance,
            Self::ImpedanceFault,
            Self::VoltageSag,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    None,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendedAction {
    Monitor,
    Derate,
    IsolateModule,
    TripBreaker,
    AlertNoc,
}

impl RecommendedAction {
    pub fn for_fault(fault: FaultClass, confidence: f32) -> Self {
        match fault {
            FaultClass::Normal => Self::Monitor,
            FaultClass::CellImbalance if confidence < 0.7 => Self::Derate,
            FaultClass::CellImbalance => Self::IsolateModule,
            FaultClass::VoltageSag => Self::Derate,
            FaultClass::ImpedanceFault => Self::IsolateModule,
            FaultClass::ThermalRunaway if confidence >= 0.85 => Self::TripBreaker,
            FaultClass::ThermalRunaway => Self::AlertNoc,
        }
    }
}
