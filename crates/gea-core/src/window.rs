use crate::TelemetrySample;

/// Sliding window of telemetry used for feature extraction.
#[derive(Debug, Clone)]
pub struct FeatureWindow {
    pub samples: Vec<TelemetrySample>,
}

impl FeatureWindow {
    pub fn new(samples: Vec<TelemetrySample>) -> Self {
        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Eight-dimensional feature vector for edge inference.
    pub fn features(&self) -> [f32; 8] {
        if self.samples.is_empty() {
            return [0.0; 8];
        }

        let n = self.samples.len() as f32;
        let last = self.samples.last().unwrap();
        let first = self.samples.first().unwrap();

        let mean_voltage = self.samples.iter().map(|s| s.voltage_v).sum::<f32>() / n;
        let mean_temp = self.samples.iter().map(|s| s.temperature_c).sum::<f32>() / n;
        let mean_impedance = self.samples.iter().map(|s| s.impedance_mohm).sum::<f32>() / n;

        let max_temp = self
            .samples
            .iter()
            .map(|s| s.temperature_c)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_voltage = self
            .samples
            .iter()
            .map(|s| s.voltage_v)
            .fold(f32::INFINITY, f32::min);

        let dv_dt = (last.voltage_v - first.voltage_v) / n;
        let dt_dt = (last.temperature_c - first.temperature_c) / n;

        let voltage_spread = {
            let max_v = self
                .samples
                .iter()
                .map(|s| s.voltage_v)
                .fold(f32::NEG_INFINITY, f32::max);
            max_v - min_voltage
        };

        [
            mean_voltage,
            mean_temp,
            mean_impedance,
            max_temp,
            min_voltage,
            dv_dt,
            dt_dt,
            voltage_spread,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn sample(v: f32, t: f32, z: f32) -> TelemetrySample {
        TelemetrySample {
            sample_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            rack_id: "rack-a1".into(),
            module_index: 0,
            voltage_v: v,
            current_a: 50.0,
            temperature_c: t,
            impedance_mohm: z,
            state_of_charge_pct: 80.0,
            label: None,
        }
    }

    #[test]
    fn extracts_eight_features() {
        let window = FeatureWindow::new(vec![
            sample(3.6, 25.0, 1.2),
            sample(3.55, 26.0, 1.3),
            sample(3.5, 28.0, 1.5),
        ]);
        let f = window.features();
        assert_eq!(f.len(), 8);
        assert!(f[6] > 0.0); // rising temperature
        assert!(f[5] < 0.0); // falling voltage
    }
}
