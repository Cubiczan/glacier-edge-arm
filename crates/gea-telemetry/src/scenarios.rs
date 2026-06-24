use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultScenario {
    Normal,
    ThermalRunaway,
    CellImbalance,
    ImpedanceFault,
    VoltageSag,
}

impl FaultScenario {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::ThermalRunaway => "thermal_runaway",
            Self::CellImbalance => "cell_imbalance",
            Self::ImpedanceFault => "impedance_fault",
            Self::VoltageSag => "voltage_sag",
        }
    }

    pub fn all() -> &'static [FaultScenario] {
        &[
            Self::Normal,
            Self::ThermalRunaway,
            Self::CellImbalance,
            Self::ImpedanceFault,
            Self::VoltageSag,
        ]
    }
}
