use chrono::{Duration, Utc};
use gea_core::{FaultClass, RackTelemetry, TelemetrySample};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use uuid::Uuid;

use crate::scenarios::FaultScenario;

#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    pub rack_id: String,
    pub module_count: u8,
    pub window_size: usize,
    pub seed: u64,
    /// Nominal cell voltage (ESS rack module, ~48V string segments mapped to per-cell proxy)
    pub nominal_voltage_v: f32,
    pub nominal_temp_c: f32,
    pub nominal_impedance_mohm: f32,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            rack_id: "rack-dc-a1".into(),
            module_count: 8,
            window_size: 16,
            seed: 42,
            nominal_voltage_v: 3.65,
            nominal_temp_c: 28.0,
            nominal_impedance_mohm: 1.2,
        }
    }
}

pub struct TelemetryGenerator {
    config: GeneratorConfig,
    rng: ChaCha8Rng,
}

impl TelemetryGenerator {
    pub fn new(config: GeneratorConfig) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(config.seed);
        Self { config, rng }
    }

    pub fn config(&self) -> &GeneratorConfig {
        &self.config
    }

    /// Generate a labeled window for one module under the given scenario.
    pub fn generate_window(
        &mut self,
        module_index: u8,
        scenario: FaultScenario,
    ) -> Vec<TelemetrySample> {
        let label = match scenario {
            FaultScenario::Normal => FaultClass::Normal,
            FaultScenario::ThermalRunaway => FaultClass::ThermalRunaway,
            FaultScenario::CellImbalance => FaultClass::CellImbalance,
            FaultScenario::ImpedanceFault => FaultClass::ImpedanceFault,
            FaultScenario::VoltageSag => FaultClass::VoltageSag,
        };

        let base_time = Utc::now();
        let mut samples = Vec::with_capacity(self.config.window_size);

        for step in 0..self.config.window_size {
            let t = step as f32 / self.config.window_size as f32;
            let noise_v = self.rng.gen_range(-0.01..0.01);
            let noise_temp = self.rng.gen_range(-0.3..0.3);
            let noise_z = self.rng.gen_range(-0.05..0.05);

            let (voltage_v, temperature_c, impedance_mohm) = match scenario {
                FaultScenario::Normal => (
                    self.config.nominal_voltage_v + noise_v,
                    self.config.nominal_temp_c + noise_temp,
                    self.config.nominal_impedance_mohm + noise_z,
                ),
                FaultScenario::ThermalRunaway => (
                    self.config.nominal_voltage_v - 0.15 * t + noise_v,
                    self.config.nominal_temp_c + 18.0 * t + noise_temp,
                    self.config.nominal_impedance_mohm + 0.4 * t + noise_z,
                ),
                FaultScenario::CellImbalance => (
                    self.config.nominal_voltage_v - 0.22 * t + noise_v,
                    self.config.nominal_temp_c + 3.0 * t + noise_temp,
                    self.config.nominal_impedance_mohm + noise_z,
                ),
                FaultScenario::ImpedanceFault => (
                    self.config.nominal_voltage_v - 0.05 * t + noise_v,
                    self.config.nominal_temp_c + 4.0 * t + noise_temp,
                    self.config.nominal_impedance_mohm + 2.5 * t + noise_z,
                ),
                FaultScenario::VoltageSag => (
                    self.config.nominal_voltage_v - 0.42 * t + noise_v,
                    self.config.nominal_temp_c + 1.0 * t + noise_temp,
                    self.config.nominal_impedance_mohm + noise_z,
                ),
            };

            samples.push(TelemetrySample {
                sample_id: Uuid::new_v4(),
                timestamp: base_time + Duration::milliseconds(step as i64 * 100),
                rack_id: self.config.rack_id.clone(),
                module_index,
                voltage_v,
                current_a: 45.0 + self.rng.gen_range(-2.0..2.0),
                temperature_c,
                impedance_mohm,
                state_of_charge_pct: 75.0 + self.rng.gen_range(-1.0..1.0),
                label: Some(label),
            });
        }

        samples
    }

    /// Full rack snapshot: one window per module for the scenario.
    pub fn generate_rack(&mut self, scenario: FaultScenario) -> RackTelemetry {
        let mut samples = Vec::new();
        for module in 0..self.config.module_count {
            samples.extend(self.generate_window(module, scenario));
        }
        RackTelemetry {
            rack_id: self.config.rack_id.clone(),
            samples,
        }
    }

    /// Mixed dataset for evaluation: equal windows per scenario.
    pub fn generate_dataset(&mut self, windows_per_scenario: usize) -> Vec<Vec<TelemetrySample>> {
        let mut dataset = Vec::new();
        for scenario in FaultScenario::all() {
            for _ in 0..windows_per_scenario {
                let module = self.rng.gen_range(0..self.config.module_count);
                dataset.push(self.generate_window(module, *scenario));
            }
        }
        dataset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_runaway_raises_temperature() {
        let mut gen = TelemetryGenerator::new(GeneratorConfig::default());
        let window = gen.generate_window(0, FaultScenario::ThermalRunaway);
        let first = window.first().unwrap().temperature_c;
        let last = window.last().unwrap().temperature_c;
        assert!(last > first + 10.0);
    }

    #[test]
    fn dataset_has_all_scenarios() {
        let mut gen = TelemetryGenerator::new(GeneratorConfig::default());
        let dataset = gen.generate_dataset(3);
        assert_eq!(dataset.len(), 15);
    }
}
