use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use gea_core::FeatureWindow;
use gea_governance::{default_ledger_path, AuditLedger, ChpPolicy, GovernanceConfig};
use gea_inference::{run_benchmark, EdgeClassifier, EvaluationReport, QuantizedClassifier};
use gea_telemetry::{FaultScenario, GeneratorConfig, TelemetryGenerator};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "gea",
    about = "GlacierEdge-Arm — Arm-optimized BESS fault detection for data center backup power",
    version
)]
struct Cli {
    #[arg(long, default_value = ".")]
    root: PathBuf,

    #[arg(long, default_value = "policies/chp.yaml")]
    policy: PathBuf,

    #[arg(long, env = "GEA_SIGNING_KEY")]
    signing_key: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate synthetic BESS telemetry JSON
    Generate {
        #[arg(long, default_value = "normal")]
        scenario: String,
        #[arg(long, default_value = "data/telemetry.json")]
        output: PathBuf,
        #[arg(long, default_value_t = 16)]
        window_size: usize,
    },
    /// Run edge inference on a telemetry JSON file
    Infer {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "int8")]
        precision: String,
    },
    /// Benchmark FP32 vs INT8 inference kernels
    Bench {
        #[arg(long, default_value_t = 10_000)]
        iterations: u64,
        #[arg(long)]
        json: bool,
    },
    /// Evaluate classifier accuracy on synthetic dataset
    Evaluate {
        #[arg(long, default_value_t = 50)]
        windows_per_scenario: usize,
    },
    /// Demo: inject fault scenario and show detection + audit trail
    Demo {
        #[arg(long, default_value = "thermal_runaway")]
        scenario: String,
    },
    /// Show CHP governance policy
    Policy,
    /// Show signed inference audit trail
    Audit,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let policy_path = resolve_path(&cli.root, &cli.policy);

    match cli.command {
        Commands::Policy => {
            let policy = if policy_path.exists() {
                ChpPolicy::load(&policy_path).map_err(anyhow::Error::msg)?
            } else {
                ChpPolicy::default()
            };
            println!("{}", serde_yaml::to_string(&policy)?);
        }
        Commands::Generate {
            scenario,
            output,
            window_size,
        } => {
            let fault = parse_scenario(&scenario)?;
            let config = GeneratorConfig {
                window_size,
                ..Default::default()
            };
            let mut gen = TelemetryGenerator::new(config);
            let window = gen.generate_window(0, fault);
            write_json(&output, &window)?;
            println!("Wrote {} samples to {}", window.len(), output.display());
        }
        Commands::Infer { input, precision } => {
            let samples: Vec<gea_core::TelemetrySample> =
                read_json(&input).context("parse telemetry JSON")?;
            let window = FeatureWindow::new(samples);
            let result = match precision.as_str() {
                "fp32" => EdgeClassifier::new().infer_window(&window),
                _ => QuantizedClassifier::new().infer_window(&window),
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Commands::Bench { iterations, json } => {
            let report = run_benchmark(iterations);
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_bench_report(&report);
            }
        }
        Commands::Evaluate {
            windows_per_scenario,
        } => {
            let mut gen = TelemetryGenerator::new(GeneratorConfig::default());
            let dataset = gen.generate_dataset(windows_per_scenario);
            let report = gea_inference::evaluate::evaluate_windows(&dataset);
            print_eval_report(&report);
        }
        Commands::Demo { ref scenario } => {
            run_demo(&cli, &policy_path, scenario)?;
        }
        Commands::Audit => {
            let ledger = ledger(&cli);
            let events = ledger.read_all()?;
            if events.is_empty() {
                println!("No audit events recorded.");
            } else {
                for event in events {
                    println!("{}", serde_json::to_string(&event)?);
                }
            }
        }
    }

    Ok(())
}

fn run_demo(cli: &Cli, policy_path: &Path, scenario: &str) -> Result<()> {
    let fault = parse_scenario(scenario)?;
    let mut gen = TelemetryGenerator::new(GeneratorConfig::default());
    let window = gen.generate_window(0, fault);
    let feature_window = FeatureWindow::new(window);

    let policy = if policy_path.exists() {
        ChpPolicy::load(policy_path).map_err(anyhow::Error::msg)?
    } else {
        ChpPolicy::default()
    };

    let clf = QuantizedClassifier::new();
    let inference = clf.infer_window(&feature_window);

    let ledger = ledger(cli);
    let policy_passed = ledger.passes_policy(&inference)
        && inference.confidence >= policy.inference_confidence_floor;
    let audit = ledger.record_inference(&inference, policy_passed)?;

    println!("=== GlacierEdge-Arm Demo ===");
    println!("Scenario:      {}", scenario);
    println!("Rack:          {}", inference.rack_id);
    println!("Module:        {}", inference.module_index);
    println!(
        "Detected:      {} ({:.1}% confidence)",
        inference.fault_class.as_str(),
        inference.confidence * 100.0
    );
    println!("Severity:      {:?}", inference.severity);
    println!("Action:        {:?}", inference.recommended_action);
    println!(
        "Latency:       {} µs ({})",
        inference.latency_us, inference.precision
    );
    println!("Policy passed: {}", policy_passed);
    println!("Audit event:   {}", audit.trace.id);
    Ok(())
}

fn ledger(cli: &Cli) -> AuditLedger {
    AuditLedger::new(GovernanceConfig {
        signing_key: cli.signing_key.clone(),
        ledger_path: default_ledger_path(&cli.root),
        confidence_floor: 0.55,
    })
}

fn parse_scenario(s: &str) -> Result<FaultScenario> {
    match s {
        "normal" => Ok(FaultScenario::Normal),
        "thermal_runaway" => Ok(FaultScenario::ThermalRunaway),
        "cell_imbalance" => Ok(FaultScenario::CellImbalance),
        "impedance_fault" => Ok(FaultScenario::ImpedanceFault),
        "voltage_sag" => Ok(FaultScenario::VoltageSag),
        other => anyhow::bail!("unknown scenario: {other}"),
    }
}

fn resolve_path(root: &Path, path: &Path) -> PathBuf {
    if path.is_relative() {
        root.join(path)
    } else {
        path.to_path_buf()
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<T> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn write_json<T: serde::Serialize>(path: &PathBuf, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    writeln!(file, "{}", serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn print_bench_report(report: &gea_inference::BenchReport) {
    println!("=== GlacierEdge-Arm Benchmark ({}) ===", report.platform);
    println!(
        "FP32:  mean {:.2} µs | p99 {} µs | model {} bytes",
        report.fp32.mean_us, report.fp32.p99_us, report.fp32.model_bytes
    );
    println!(
        "INT8:  mean {:.2} µs | p99 {} µs | model {} bytes",
        report.int8.mean_us, report.int8.p99_us, report.int8.model_bytes
    );
    println!(
        "Speedup: {:.2}x | Size reduction: {:.2}x",
        report.speedup_ratio, report.size_reduction_ratio
    );
}

fn print_eval_report(report: &EvaluationReport) {
    println!("=== Evaluation Report ===");
    println!(
        "Accuracy: {:.1}% ({}/{})",
        report.accuracy * 100.0,
        report.correct,
        report.windows_evaluated
    );
    println!(
        "False positive rate: {:.1}%",
        report.false_positive_rate * 100.0
    );
    for class in &report.per_class {
        if class.support > 0 {
            println!(
                "  {}: recall {:.1}% ({} / {})",
                class.fault_class,
                class.recall * 100.0,
                class.correct,
                class.support
            );
        }
    }
}
